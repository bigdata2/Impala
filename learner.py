"""Learner with parameter server"""

import argparse
import random
import numpy as np
import os

import deepmind_lab
from model import model_A3C
import utils
from parameterserver import ParameterServer
from actor import Actor
import ray
import torch
import torch.nn.functional as F

ACTION_LIST = utils.getactions().values()
 
@ray.remote(num_gpus=1)
class Learner(object):
  """Learner to get trajectories from Actors running DeepMind Lab simulator."""

  def __init__(self, ps):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0] % 4)
    print("Initialize learner environment gpu id: ", os.environ["CUDA_VISIBLE_DEVICES"])
    self.id = -1 
    self.parameterserver = ps
    self.model = model_A3C()
    params = self.model.cpu().state_dict()
    self.parameterserver.push.remote(dict(params))
    self.model = self.model.cuda()
    self.lr = 1e-5
    self.wd = 1e-3
    self.eps = 1e-5
  
  def get_id(self):
    return self.id

  def run(self, length, width, height, fps, level, record, demo, video, 
          agents_num, actors, gamma):
    """Gets trajectories from actors and trains learner."""
    print("level............................... ", level)
    self.gamma = gamma
    config = {
        'fps': str(fps),
        'width': str(width),
        'height': str(height)
    }
    if record:
      config['record'] = record
    if demo:
      config['demo'] = demo
    if video:
      config['video'] = video
    config['demofiles'] = "/tmp"
    testactor = actors.pop()
    actorsObjIds = [actor.run_train.remote() for actor in actors]
    testactorsObjId = [testactor.run_test.remote()]
    actorsObjIds += testactorsObjId
    optimizer = self.create_optimizer()
    queue = []
    policy_loss, value_loss = None, None
    while True:
    	ready, actorsObjIds = ray.wait(actorsObjIds, 1)
   	trajectory = ray.get(ready)
	if not trajectory[0]:
    	    actorsObjIds.extend([testactor.run_test.remote()])
	    print("policy_loss, value_loss ", policy_loss, value_loss)
	    continue
	actorsObjIds.extend([actors[trajectory[0].actor_id].run_train.remote()])
	queue.append(trajectory[0])
	if len(queue) < 4: continue #batch size of 4
	self.model.zero_grad()
	for t in queue:
		policy_loss, value_loss = self.train(t, optimizer)
	optimizer.step()
    	params = self.model.cpu().state_dict()
    	self.parameterserver.push.remote(dict(params))
    	self.model = self.model.cuda()
	queue = []
    return

  def create_optimizer(self):
        # setup optimizer
        #optimizer = torch.optim.Adam(self.model.parameters(), self.lr,
        #                           weight_decay=self.wd)
	optimizer = torch.optim.RMSprop(self.model.parameters(), 
		    lr=self.lr, eps=self.eps)
        return optimizer

  def clipreward(self, reward):
	tanh = torch.nn.Tanh()
	reward = tanh(torch.FloatTensor([reward]).cuda())
	reward =  0.3 * min(reward, 0) + 5.0 * max(reward, 0)
	return reward
    
  def train(self, trajectory, optimizer):
	if trajectory.length() < 2: return None, None
	states_batch = utils.createbatch(trajectory.states)
	fc_out = self.model(states_batch)
	hin = torch.cuda.FloatTensor(trajectory.lstm_hin)
	cin = torch.cuda.FloatTensor(trajectory.lstm_cin)
	lstm_out = []
	for i in reversed(range(trajectory.length())):
    	# Step through the convnet+fc-out one state at a time.
	    lstm_in = fc_out[i].unsqueeze(0)
    	    hin, cin = self.model.lstm(lstm_in, (hin,cin))
	    lstm_out += [hin]
	lstm_out_tensor = torch.stack(lstm_out)
	actions = self.model.actor_linear(lstm_out_tensor)
	values = self.model.critic_linear(lstm_out_tensor)
	action_prob = self.model.softmax(actions)
	action_log_prob = F.log_softmax(actions)
	entropy = -(action_log_prob * action_prob).sum(2)
	R = torch.cuda.FloatTensor(0) if trajectory.terminal else values[-1][0]
	value_loss = 0
	policy_loss = 0
	for i in reversed(range(trajectory.length()-1)):
            R = self.gamma * R + self.clipreward(trajectory.rewards[i])
            advantage = R - values[i][0]
            value_loss = value_loss + 0.5 * advantage.pow(2)
	    mu_idx = trajectory.actions[i] 
	    importance_weight = action_prob[i][0][mu_idx] / \
			        torch.cuda.FloatTensor([trajectory.pi_at_st[i]])
	    importance_weight = torch.clamp(importance_weight, max=1.0)

	    policy_loss = policy_loss - \
			  importance_weight * \
                	  action_log_prob[i][0][mu_idx] * \
			  advantage - \
                	  0.01 * entropy[i]
        (policy_loss + 0.5 * value_loss).backward()
        return policy_loss, value_loss

if __name__ == '__main__':
   RAY_HEAD="10.145.142.25:6379"
   NUMBER_OF_ACTORS=5
   
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("-s", "--standalone",
                       help="run the program with stand alone ray (no cluster)", action="store_true")
   
   parser.add_argument("--cluster",
                       help="the address of the head of the cluster, default is {0}".format(RAY_HEAD), default=RAY_HEAD)
   parser.add_argument("--actors", type=int, default=NUMBER_OF_ACTORS,
                       help="the number of actors to start, default is {0}".format(NUMBER_OF_ACTORS))
   parser.add_argument("--gamma", type=float, default=0.99,
                       help="the discount factor, default is {0}".format(0.99))
   parser.add_argument('--length', type=int, default=1000,
                       help='Number of steps to run the agent')
   parser.add_argument('--width', type=int, default=280,
                       help='Horizontal size of the observations')
   parser.add_argument('--height', type=int, default=280,
                       help='Vertical size of the observations')
   parser.add_argument('--fps', type=int, default=60,
                       help='Number of frames per second')
   parser.add_argument('--savemodel_path', type=str, default="./checkpoint.pt",
                       help='Set the path to save trained model parameters')
   parser.add_argument('--loadmodel_path', type=str, default="./checkpoint.pt",
                       help='Set the path to load trained model parameters')
   parser.add_argument('--test', type=str, default=None,
                       help='Train or Test')
   parser.add_argument('--runfiles_path', type=str, default=None,
                       help='Set the runfiles path to find DeepMind Lab data')
   parser.add_argument('--level_script', type=str,
                       #default='tests/empty_room_test',
                       default='stairway_to_melon',
                       help='The environment level script to load')
   parser.add_argument('--record', type=str, default=None,#"record",
                       help='Record the run to a demo file')
   parser.add_argument('--demo', type=str, default=None,#"record",#None,
                       help='Play back a recorded demo file')
   parser.add_argument('--video', type=str, default=None,#'testvideo',#None,
                       help='Record the demo run as a video')
 
   args = parser.parse_args()
 
   print("Using Ray Cluster on {}".format(args.cluster))
   if args.standalone is True:
     ray.init(num_gpus=32)
   else:
     ray.init(redis_address=args.cluster)
   
   if args.runfiles_path:
     deepmind_lab.set_runfiles_path(args.runfiles_path)

   # Start the Parameter Server.
   ps = ParameterServer.remote({})

   # Start Learner
   learner = Learner.remote(ps)

   config = {
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height)
   }
   if args.record:
     config['record'] = record
   if args.demo:
     config['demo'] = demo
   if args.video:
     config['video'] = video
   config['demofiles'] = "/tmp"

   # Start actors.
   #TODO: pass arg dictionary instead of individuals parameters
   actors = [Actor.remote(idx, args.length, args.level_script, config, ps,
	                 args.savemodel_path, args.loadmodel_path, args.test) 
	    for idx in range(args.actors)]

   #TODO: pass arg dictionary instead of individuals parameters
   objid = learner.run.remote(args.length, args.width, args.height, 
		args.fps, args.level_script, args.record, args.demo,
	        args.video, args.actors, actors, args.gamma)
  
   ray.wait([objid])
