# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Hamiltonian Neural Networks (HNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to HNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep HNN arichtecture to improve predictive performance

import torch, argparse
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#***** Names of coded probability distribution functions: *****
# - '1D_Gauss_mix'
# - '2D_Gauss_mix'
# - '5D_illconditioned_Gaussian'
# - 'nD_Funnel'
# - 'nD_Rosenbrock'
# - 'nD_standard_Gaussian'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--num_samples', default=10, type=int, help='number of training samples simulated using Hamiltonian Monte Carlo')
    parser.add_argument('--len_sample', default=50, type=int, help='length of Hamiltonian trajectory for each training sample')
    parser.add_argument('--dist_name', default='nD_standard_Gaussian', type=str, help='name of the probability distribution function')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--load_dir', default=THIS_DIR, type=str, help='where to load the training data from')
    parser.add_argument('--should_load', default=False, type=bool, help='should load training data?')
    parser.add_argument('--load_file_name', default='nD_standard_Gaussian', type=str, help='should load training data?')
    parser.add_argument('--total_steps', default=5000, type=int, help='number of gradient steps')

    parser.add_argument('--hidden_dim', default=100, type=int, help='hidden dimension of mlp')
    parser.add_argument('--num_layers', default=3, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--test_fraction', default=0., type=float, help='fraction of testing samples')
    parser.add_argument('--step_size', default=0.025, type=int, help='step size for time integration')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.set_defaults(feature=True)
    return parser.parse_args()