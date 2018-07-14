"""Actor to generate trajactories"""

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
import actions
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_LIST = actions.getactions().values()

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
    self.step     = step #trajectory number for debugging only

  def length(self):
    return len(self.rewards)
 
@ray.remote
class Actor(object):
  """Simple actor for DeepMind Lab."""

  def __init__(self, idx, length, level, config, ps):
    print("Initialize Actor environment")
    self.id = idx
    self.steps = 0
    self.parameterserver = ps
    self.length = length
    self.env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    self.env.reset()
    action_spec = self.env.action_spec()
    self.model = model_A3C(isActor=True)
    self.lstm_init = torch.zeros(1, 256) #TODO remove hardcoding
    self.cin = self.lstm_init
    self.hin = self.lstm_init
  
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
    for _ in range(self.length):
      if not self.env.is_running():
        print('Environment stopped. Restarting...')
        self.env.reset()
	self.steps = 0
    	self.cin = self.lstm_init
    	self.hin = self.lstm_init
	if rollout.length(): break
    
      obs = self.env.observations()
      logits, (self.hin, self.cin) = self.model(obs['RGB_INTERLEAVED'], self.cin, self.hin)
      prob = F.softmax(logits, dim=1)
      action_idx = prob.multinomial(1)[0].tolist()[0]
      pi = prob[0][action_idx].tolist()
      action = ACTION_LIST[action_idx]
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward
      rollout.append(obs['RGB_INTERLEAVED'], action, reward, pi, self.steps)
    print("Rollout Finished Total Reward for actor_id {}:  {}".format(self.id, totalreward))
    return rollout

  def run_test(self):
    """Run the env for n steps and return a trajectory rollout."""
    weights = ray.get(self.parameterserver.pull.remote())
    self.model.load_state_dict(weights)
    rollout = trajectory()
    rollout.actor_id = self.id
    totalreward = 0
    self.steps += 1
    for _ in range(self.length):
      if not self.env.is_running():
        print('Environment stopped. Restarting...')
        self.env.reset()
        self.steps = 0
        self.cin = self.lstm_init
        self.hin = self.lstm_init
        if rollout.length(): break

      obs = self.env.observations()
      logits, (self.hin, self.cin) = self.model(obs['RGB_INTERLEAVED'], self.cin, self.hin)
      prob = F.softmax(logits, dim=1)
      action_idx = prob.max(1)[1].tolist()[0]
      action = ACTION_LIST[action_idx]
      #print("logits.shape, self.hin.shape, self.cin.shape ", logits.shape ,self.hin.shape, self.cin.shape)
      #print("logits ", logits)
      print("prob ", prob)
      print("action ", action)
      #print("logits,self.hin, self.cin ", logits,self.hin, self.cin)
      action = random.choice(ACTION_LIST)
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward
      rollout.append(obs['RGB_INTERLEAVED'], action, reward, self.steps)
    print("Rollout Finished Total Reward for actor_id {}:  {}".format(self.id, totalreward))
    return rollout
      
  def get_id(self):
    return self.id

  def get_reward(self):
    return self.reward

  def get_steps(self):
    return self.steps
