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
 
@ray.remote
class Actor(object):
  """Simple agent for DeepMind Lab."""

  def __init__(self, idx, length, level, config, ps):
    # Set an environment variable to tell TensorFlow which GPUs to use. Note
    # that this must be done before the call to tf.Session.
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
    trajectory = []
    totalreward = 0
    steps = 0
    for _ in range(self.length):
      if not self.env.is_running():
        print('Environment stopped early')
        self.env.reset()
	break
    
      obs = self.env.observations()
      #print("State ---> {}".format(obs['RGB_INTERLEAVED']))
      self.model(obs['RGB_INTERLEAVED'])
      action = random.choice(ACTION_LIST)
      #print("Action ---> {}".format(action))      
      reward = self.env.step(action, num_steps=4) #for action repeat=4
      totalreward += reward
      #print("Reward ---> {}".format(reward))
      steps += 1
      trajectory.append((obs['RGB_INTERLEAVED'], action, reward))
    print("Rollout Finished Total Reward:  {}".format(totalreward))
    #print ("len(trajectory) ",len(trajectory))
    return trajectory
      
  def get_id(self):
    return self.id

  def get_reward(self):
    return self.reward

  def get_steps(self):
    return self.steps
