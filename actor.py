"""Agent to generate trajactories"""

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
import ray

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
  'forward': _action(0, 0, 0, 1, 0, 0, 0),
  'backward': _action(0, 0, 0, -1, 0, 0, 0),
  'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
  'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
  'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
  'look_right': _action(20, 0, 0, 0, 0, 0, 0),
  'forward_look_left': _action(-20, 0, 0, 1, 0, 0, 0),
  'forward_look_right': _action(20, 0, 0, 1, 0, 0, 0),
  'fire': _action(0, 0, 0, 0, 1, 0, 0),
}

ACTION_LIST = ACTIONS.values()


class trajectory(object):
  """class to store trajectory data."""
  
  def __init__(self):
    self.states  = []
    self.actions = []
    self.rewards = []
    self.actor_id = None
    #self.terminal = False

  def append(self, state, action, reward):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward] 
    #self.terminal = terminal

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
  
  def run(self):    
    """Gets an image state and a reward, returns an action."""
    print("Step {}".format(self.steps))

    weights = ray.get(self.parameterserver.pull.remote())
    self.model.load_state_dict(weights)
    rollout = trajectory()
    rollout.actor_id = self.id
    totalreward = 0
    steps = 0
    for _ in range(self.length):
      if not self.env.is_running():
        print('Environment stopped early')
        self.env.reset()
	break
    
      obs = self.env.observations()
      self.model(obs['RGB_INTERLEAVED'])
      action = random.choice(ACTION_LIST)
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward
      steps += 1
      rollout.append(obs['RGB_INTERLEAVED'], action, reward)
    print("Rollout Finished Total Reward:  {}".format(totalreward))
    #print ("len(rollout) ",rollout.length())
    return rollout
      
  def get_id(self):
    return self.id

  def get_reward(self):
    return self.reward

  def get_steps(self):
    return self.steps
