# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Langevin Monte Carlo with HNNs

import torch, sys
import autograd.numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow as tf
import tensorflow_probability as tfp
from nn_models import MLP
from hnn import HNN
from scipy.stats import norm
from scipy.stats import uniform
from get_args import get_args
from utils import leapfrog
from functions import functions
args = get_args()

##### User-defined sampling parameters #####

chains = 1 # number of Markov chains
N = 10000 # number of samples
epsilon = 0.025 # step size
burn = 1000 # number of burn-in samples

##### Sampling code below #####

def get_model(args, baseline):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=baseline)
    path = args.dist_name + ".tar"
    model.load_state_dict(torch.load(path))
    return model

def integrate_model(model, t_span, y0, n, **kwargs):
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

t_span = [0,epsilon]
steps = 2
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
hnn_model = get_model(args, baseline=False)
y0 = np.zeros(args.input_dim)
hnn_fin = np.zeros((chains,N,int(args.input_dim/2)))
hnn_accept = np.zeros((chains,N))
for ss in np.arange(0,chains,1):
    x_req = np.zeros((N,int(args.input_dim/2)))
    x_req[0,:] = y0[0:int(args.input_dim/2)]
    accept = np.zeros(N)
    
    for ii in np.arange(0,int(args.input_dim/2),1):
        y0[ii] = 0.0
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs()
    HNN_sto = np.zeros((args.input_dim,1,N))
    for ii in np.arange(0,N,1):
        hnn_ivp = integrate_model(hnn_model, t_span, y0, steps-1, **kwargs)
        for sss in range(0,args.input_dim):
            HNN_sto[sss,:,ii] = hnn_ivp[sss,1]
        yhamil = np.zeros(args.input_dim)
        for jj in np.arange(0,args.input_dim,1):
            yhamil[jj] = hnn_ivp[jj,1]
        H_star = functions(yhamil)
        H_prev = functions(y0)
        alpha = np.minimum(1,np.exp(H_prev - H_star))
        if alpha > uniform().rvs():
            y0[0:int(args.input_dim/2)] = hnn_ivp[0:int(args.input_dim/2),1]
            x_req[ii,:] = hnn_ivp[0:int(args.input_dim/2),1]
            accept[ii] = 1
        else:
            x_req[ii,:] = y0[0:int(args.input_dim/2)]
        for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
            y0[jj] = norm(loc=0,scale=1).rvs()
        print("Sample: "+str(ii)+" Chain: "+str(ss))
    hnn_accept[ss,:] = accept
    hnn_fin[ss,:,:] = x_req

ess_hnn = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(hnn_fin[ss,burn:N,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))