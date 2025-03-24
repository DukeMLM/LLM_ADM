import torch
import torch.nn as nn

class ModelParams:
    def __init__(self):
        self.parameters = {}

class NNParams(ModelParams):
    def __init__(self, learning_rate, num_layer, num_neuron):
        super(NNParams, self).__init__()
        self.learning_rate = learning_rate
        self.num_layer = num_layer
        self.num_neuron = num_neuron
        self.lr_decay_rate = 0.2
        self.weight_decay = 1e-4
        self.num_epoch = 300
        self.eval_step = 10

class RFParams(ModelParams):
    def __init__(self, node):
        super(RFParams, self).__init__()
        self.node = node