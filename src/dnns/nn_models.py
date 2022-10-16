# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Deep Neural Networks (DNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to DNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep DNN arichtecture to improve predictive performance

import torch, argparse
import numpy as np
from utils import choose_nonlinearity
from get_args import get_args
args = get_args()

class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    return self.linear5(h)