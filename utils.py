# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torchvision import transforms
import torch

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
            #'fire': _action(0, 0, 0, 0, 1, 0, 0),
          }

def getactions():
    return ACTIONS

def action_space():
    return len(ACTIONS)

def createbatch(img_list):
    normalize = transforms.Normalize(
                        mean = [127.5, 127.5, 127.5],
                        std  = [127.5, 127.5, 127.5]
    )
    preprocess = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((96,72)),
                        transforms.ToTensor(),
                        normalize, #TODO normalize inside the model running on GPU
			transforms.Lambda(lambda img_tensor: img_tensor.unsqueeze_(0))
    ])
     
    img_tensor = torch.zeros(0, 0)
    for i in range(len(img_list)):
    	img_tensor = torch.cat((img_tensor,preprocess(img_list[i])),0)
    #print img_tensor.shape
    return img_tensor

