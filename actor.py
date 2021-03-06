# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""

import argparse
import random
import numpy as np
import os
import time
import deepmind_lab
import pprint
from model import model_A3C
import utils
import ray
import torch
import torch.nn.functional as F


ACTION_LIST = utils.getactions().values()

class trajectory(object):
  """class to store trajectory data."""
  
  def __init__(self):
    self.states   = []
    self.actions  = []
    self.rewards  = []
    self.pi_at_st = []
    self.actor_id = None
    self.lstm_hin = None
    self.lstm_cin = None

  def append(self, state, action, reward, pi, step):
    self.states   += [state]
    self.actions  += [action]
    self.rewards  += [reward] 
    self.pi_at_st += [pi]
    self.terminal = False

  def length(self):
    return len(self.rewards)
 
@ray.remote(num_gpus=1)
class Actor(object):
  """Simple actor for DeepMind Lab."""

  def __init__(self, idx, length, level, config, ps, savemodel_path, loadmodel_path, test):
    #Running actor on fractional GPU, see https://github.com/ray-project/ray/issues/402#issuecomment-363590303
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0] % 4)
    print("Initialize Actor environment gpu id: ", os.environ["CUDA_VISIBLE_DEVICES"])
    self.id = idx
    self.steps = 0
    self.parameterserver = ps
    self.length = length
    self.env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    self.env.reset()
    action_spec = self.env.action_spec()
    self.model = model_A3C(isActor=True)
    self.model = self.model.cuda()
    self.lstm_init = torch.zeros(1, 256).cuda() #TODO remove hardcoding
    self.cin = self.lstm_init
    self.hin = self.lstm_init
    self.rewards = 0
    self.savepath = savemodel_path
    self.loadpath = loadmodel_path
    self.test = test
  
  def run_train(self):    
    """Run the env for n steps and return a trajectory rollout."""
    weights = ray.get(self.parameterserver.pull.remote())
    self.model.load_state_dict(weights)
    rollout = trajectory()
    rollout.actor_id = self.id
    totalreward = 0
    self.steps += 1
    rollout.lstm_hin = self.hin.tolist()
    rollout.lstm_cin = self.cin.tolist()
    obs, action, reward, pi = None, None, None, None
    for _ in range(self.length):
      if not self.env.is_running():
        print('Environment stopped. Restarting...')
        self.rewards = 0
        self.env.reset()
	self.steps = 0
    	self.cin = self.lstm_init
    	self.hin = self.lstm_init
        obs = self.env.observations()
        rollout.append(obs['RGB_INTERLEAVED'], action, reward, pi, self.steps)
	rollout.terminal = True
	break
    
      obs = self.env.observations()
      img_tensor = utils.createbatch([obs['RGB_INTERLEAVED']])
      prob, (self.hin, self.cin) = self.model(img_tensor, self.cin, self.hin)
      action_idx = prob.multinomial(1)[0].tolist()[0]
      pi = prob[0][action_idx].tolist()
      action = ACTION_LIST[action_idx]
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward
      action = action_idx #ACTION_LIST[action_idx]
      rollout.append(obs['RGB_INTERLEAVED'], action, reward, pi, self.steps)
    self.rewards += totalreward
    return rollout

  def run_test(self):
    """Run the env for n steps and return a trajectory rollout."""
    time.sleep(30)
    weights = None
    if not self.test:
    	weights = ray.get(self.parameterserver.pull.remote())
        torch.save(weights, self.savepath)
    else: 
	weights = torch.load(self.loadpath,
    		             map_location=lambda storage, loc: storage)
    self.model.load_state_dict(weights)
    totalreward = 0
    self.steps += 1
    self.env.reset()
    self.cin = self.lstm_init
    self.hin = self.lstm_init
    while self.env.is_running():
      obs = self.env.observations()
      img_tensor = utils.createbatch([obs['RGB_INTERLEAVED']])
      prob, (self.hin, self.cin) = self.model(img_tensor, self.cin, self.hin)
      action_idx = prob.max(1)[1].tolist()[0]
      action = ACTION_LIST[action_idx]
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward

    print("TEST ACTOR: Test Finished Total Reward for actor_id {}:  {}".format(self.id, totalreward))
    return
      
  def get_id(self):
    return self.id

  def get_reward(self):
    return self.reward

  def get_steps(self):
    return self.steps
