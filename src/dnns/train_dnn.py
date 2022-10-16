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

from numbers import Real
import torch, argparse
import numpy as np
import os, sys
from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss, to_pickle
from get_args import get_args

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=True)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.)

  # arrange data
  data = get_dataset(seed=args.seed)
  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
  
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dcoords'])
  test_dxdt = torch.Tensor(data['test_dcoords'])

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  
  for step in range(args.total_steps+1):

    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.time_derivative(x[ixs])
    loss = L2_loss(dxdt[ixs], dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()

    # run test data
    test_dxdt_hat = model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)
    
    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    #if args.verbose and step % args.print_every == 0:
    print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2 
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    path = '{}/{}.tar'.format(args.save_dir, args.dist_name)
    torch.save(model.state_dict(), path)