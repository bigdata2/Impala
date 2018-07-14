import numpy as np

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

def getactions():
	return ACTIONS

def action_space():
	return len(ACTIONS)
