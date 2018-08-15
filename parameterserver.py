# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ray

@ray.remote
class ParameterServer(object):
    def __init__(self, weights):
        self.weights = dict(weights)

    def push(self, weights):
        self.weights = weights

    def pull(self):
        return dict(self.weights)
