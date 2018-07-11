import ray

@ray.remote
class ParameterServer(object):
    def __init__(self, weights):
        self.weights = dict(weights)

    def push(self, weights):
        self.weights = weights

    def pull(self):
        return dict(self.weights)
