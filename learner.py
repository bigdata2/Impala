"""Learner with parameter server"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import os

import deepmind_lab
import pprint
from model import model_A3C
from parameterserver import ParameterServer
from actor import Actor
import ray

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
  'look_left': _action(-20, 0, 0, 1, 0, 0, 0),
  'look_right': _action(20, 0, 0, 1, 0, 0, 0),
}

ACTION_LIST = ACTIONS.values()
 
@ray.remote(num_gpus=1)
#@ray.remote
class Learner(object):
  """Learner to get trajectories from Actors running DeepMind Lab simulator."""

  def __init__(self, ps):
    # Set an environment variable to tell TensorFlow which GPUs to use. Note
    # that this must be done before the call to tf.Session.
    print("Initialize GPU environment")
    gpu_ids = ",".join([str(i) for i in ray.get_gpu_ids()])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    self.id = -1 
    self.parameterserver = ps
    self.model = model_A3C()
    self.model = self.model.cuda()
    params = self.model.cpu().state_dict()
    self.parameterserver.push.remote(dict(params))
    print("Learner ID {} possibly on GPUs {} start ...".format(self.id, gpu_ids))
  
  def get_id(self):
    return self.id

  def run(self, length, width, height, fps, level, record, demo, video, agents_num, actors):
    """Spins up an environment and runs the random agent."""
    print("level............................... ", level)
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

    actorsObjIds = [actor.run.remote() for actor in actors]
    #ray.wait(actorsObjIds, len(agents))
    ready, _ = ray.wait(actorsObjIds, 1)
    trajectory = ray.get(ready)
    print("length of trajectory ", len(trajectory[0]))
    print(trajectory)
    

if __name__ == '__main__':
   RAY_HEAD="10.145.142.25:6379"
   NUMBER_OF_AGENTS=1
   
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("-s", "--standalone",
                       help="run the program with stand alone ray (no cluster)", action="store_true")
   
   parser.add_argument("--cluster",
                       help="the address of the head of the cluster, default is {0}".format(RAY_HEAD), default=RAY_HEAD)
   parser.add_argument("--agents", type=int, default=NUMBER_OF_AGENTS,
                       help="the number of agents to start, default is {0}".format(NUMBER_OF_AGENTS))
   parser.add_argument('--length', type=int, default=1000,
                       help='Number of steps to run the agent')
   parser.add_argument('--width', type=int, default=280,
                       help='Horizontal size of the observations')
   parser.add_argument('--height', type=int, default=280,
                       help='Vertical size of the observations')
   parser.add_argument('--fps', type=int, default=60,
                       help='Number of frames per second')
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
   print (ACTIONS['look_left'])
 
   print("Using Ray Cluster on {}".format(args.cluster))
   if args.standalone is True:
     ray.init()
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

   # Start agents.
   actors = [Actor.remote(idx, args.length, args.level_script, config, ps) for idx in range(args.agents)]

   objid = learner.run.remote(args.length, args.width, args.height, args.fps, args.level_script,
       args.record, args.demo, args.video, args.agents, actors)
  
   ray.wait([objid])
