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

import csv
import torch, argparse
import autograd.numpy as np
import autograd
from scipy.stats import norm
from functions import functions
from utils import leapfrog, to_pickle, from_pickle
from get_args import get_args
args = get_args()

def dynamics_fn(t, coords):
    dcoords = autograd.grad(functions)(coords)
    dic1 = np.split(dcoords,args.input_dim)
    S = np.concatenate([dic1[int(args.input_dim/2)]])
    for ii in np.arange(int(args.input_dim/2)+1,args.input_dim,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,int(args.input_dim/2),1):
        S = np.concatenate([S, -dic1[ii]])
    return S

def get_trajectory(t_span=[0,args.len_sample], timescale=args.len_sample, y0=None, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if y0 is None:
        y0 = np.zeros(args.input_dim)
        for ii in np.arange(0,int(args.input_dim/2),1):
            y0[ii] = norm(loc=0,scale=1).rvs()
    lp_ivp = leapfrog(dynamics_fn, t_span, y0,int(timescale*(t_span[1]-t_span[0])), args.input_dim)
    dic1 = np.split(lp_ivp, args.input_dim)
    dydt = [dynamics_fn(None, lp_ivp[:,ii]) for ii in range(0, lp_ivp.shape[1])]
    dydt = np.stack(dydt).T
    ddic1 = np.split(dydt, args.input_dim)
    return dic1, ddic1, t_eval

def get_dataset(seed=0, samples=args.num_samples, test_split=(1.0-args.test_fraction), **kwargs):
    
    if args.should_load:
        path = '{}/{}.pkl'.format(args.load_dir, args.load_file_name)
        data = from_pickle(path)
        print("Successfully loaded data")
    else:
        data = {'meta': locals()}
        # randomly sample inputs
        np.random.seed(seed) #
        xs, dxs = [], []
        index1 = 0

        count1 = 0
        y_init = np.zeros(args.input_dim)
        for ii in np.arange(0,int(args.input_dim/2),1):
            y_init[ii] = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            y_init[ii] = norm(loc=0,scale=1).rvs()

        print('Generating HMC samples for HNN training')

        for s in range(samples):
            print('Sample number ' + str(s+1) + ' of ' + str(samples))
            dic1, ddic1, t = get_trajectory(y0=y_init,**kwargs)
            xs.append(np.stack( [dic1[ii].T.reshape(len(dic1[ii].T)) for ii in np.arange(0,args.input_dim,1)]).T)
            dxs.append(np.stack( [ddic1[ii].T.reshape(len(ddic1[ii].T)) for ii in np.arange(0,args.input_dim,1)]).T)
            y_init = np.zeros(args.input_dim)
            count1 = count1 + 1
            for ii in np.arange(0,int(args.input_dim/2),1):
                y_init[ii] = dic1[ii].T[len(dic1[ii].T)-1]
            for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
                y_init[ii] = norm(loc=0,scale=1).rvs()

        data['coords'] = np.concatenate(xs)
        data['dcoords'] = np.concatenate(dxs).squeeze()

        # make a train/test split
        split_ix = int(len(data['coords']) * test_split)
        split_data = {}
        for k in ['coords', 'dcoords']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data

        # save data
        path = '{}/{}.pkl'.format(args.save_dir, args.dist_name)
        to_pickle(data, path)

    return data