# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Langevin Monte Carlo, Hamiltonian Monte Carlo, and NUTS with online error monitoring with SympNets

import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from utils import to_pickle, from_pickle
from Sampling import LMC
from Sampling import HMC
from Sampling import NUTS
import tensorflow as tf
import torch
import random
import tensorflow_probability as tfp
from get_args import get_args
args = get_args()

# We fix the seeds to ensure that the results can be reproduced
tf.random.set_seed(11)
torch.manual_seed(13)
np.random.seed(17)
random.seed(19)

## Load data and train SympNet (LA or G)

path = '{}/{}.pkl'.format(args.load_dir, args.load_file_name)
data_raw = from_pickle(path)

print("Successfully loaded data")
d1 = data_raw['coords']

# We flip the (q,p) blocks because SympNets expect the (p,q) ordering, this will be corrected within
# the samplers
xt = np.zeros((int(args.len_sample * args.num_samples), args.input_dim))
yt = np.zeros((int(args.len_sample * args.num_samples), args.input_dim))
rn = np.arange(0,args.num_samples+1) + np.arange(0,int(args.len_sample * args.num_samples)+1,int(args.len_sample))
for ii in range(args.num_samples):
    for jj in range(int(args.input_dim/2)):
        xt[int(args.len_sample)*ii:int(args.len_sample)*(ii+1),int(args.input_dim/2)+jj] = d1[rn[ii]:rn[ii+1]-1,jj]
        yt[int(args.len_sample)*ii:int(args.len_sample)*(ii+1),int(args.input_dim/2)+jj] = d1[rn[ii]+1:rn[ii+1],jj]
    for jj in range(int(args.input_dim/2)):
        xt[int(args.len_sample)*ii:int(args.len_sample)*(ii+1),jj] = d1[rn[ii]:rn[ii+1]-1,int(args.input_dim/2)+jj]
        yt[int(args.len_sample)*ii:int(args.len_sample)*(ii+1),jj] = d1[rn[ii]+1:rn[ii+1],int(args.input_dim/2)+jj]

class PDData(ln.Data):
    def __init__(self, add_h=False):
        super(PDData, self).__init__()
        self.h = args.step_size
        self.__init_data()

    @property
    def dim(self):
        return args.input_dim

    def __init_data(self):
            self.X_train = xt
            self.y_train = yt
            self.X_test = xt
            self.y_test = yt

device = 'cpu'
h = args.step_size
net_type = args.net_type
LAlayers = args.LAnum_layers
LAsublayers = args.LAnum_sub_layers
Glayers = args.Gnum_layers
Gwidth = args.Ghidden_dim
activation = args.nonlinearity
lr = args.learn_rate
iterations = args.total_steps
print_every = args.print_every

add_h = False
criterion = 'MSE'
data = PDData()
net = None
if args.read_net:
    net = torch.load(args.net_folder+"/model_best.pkl")
else:
    if net_type == 'LA':
        net = ln.nn.LASympNet(data.dim, LAlayers, LAsublayers, activation)
    elif net_type == 'G':
        net = ln.nn.GSympNet(data.dim, Glayers, Gwidth, activation)
args1 = {
    'data': data,
    'net': net,
    'criterion': criterion,
    'optimizer': 'adam',
    'lr': lr,
    'iterations': iterations,
    'batch_size': args.batch_size,
    'print_every': print_every,
    'save': True,
    'callback': None,
    'dtype': 'float',
    'device': device
}
ln.Brain.Init(**args1)

if args.train_net:
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()

## Sampling parameters

req_samples_hmc = 25000
req_samples_nuts = 100000
req_samples_lmc = 500000

hmc_len = 500
burn_in = 100

## Sampling

# Langevin Monte Carlo
lmc_samples, lmc_accept = LMC(net,req_samples_lmc)

## Compute effective sample size
hnn_tf = tf.convert_to_tensor(lmc_samples[burn_in:req_samples_lmc,:])
ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
## We need N+1 gradient evaluations in N leapfrog steps, here we have only one step so
## we need 2 evaluations (hence the factor 2)
print("ESS with LMC:", ess_hnn)
print("ESS/(number of gradient evaluation) with LMC:", ess_hnn/(2*(req_samples_lmc-burn_in)))

if args.plot_samples:
    fig, ax = plt.subplots(figsize =(10, 7))
    markevery = max(int(req_samples_lmc/15000),1)
    ax.scatter(lmc_samples[::markevery,0], lmc_samples[::markevery,1], marker='+')
    plt.savefig('lmc.pdf')

# Hamiltonian Monte Carlo
hmc_samples, hmc_accept = HMC(net,req_samples_hmc,steps = hmc_len)

## Compute effective sample size
hnn_tf = tf.convert_to_tensor(hmc_samples[burn_in:req_samples_hmc,:])
ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
## We need N+1 gradient evaluations in N leapfrog steps
print("ESS with HMC:", ess_hnn)
print("ESS/(number of gradient evaluation) with HMC:", ess_hnn/((req_samples_hmc-burn_in)*(hmc_len+1)))

if args.plot_samples:
    fig, ax = plt.subplots(figsize =(10, 7))
    markevery = max(int(req_samples_hmc/15000),1)
    ax.scatter(hmc_samples[::markevery,0], hmc_samples[::markevery,1], marker='+')
    plt.savefig('hmc.pdf')

# No-U-Turn Sampling with online error monitoring
nuts_samples, nuts_err, nuts_ind, nuts_traj, both_directions = NUTS(net,req_samples_nuts)

## Compute effective sample size
hnn_tf = tf.convert_to_tensor(nuts_samples[burn_in:req_samples_nuts,:])
ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
## We need N+1 gradient evaluations in N leapfrog steps if we go in one direction only,
## but if the tree builds in both directions the number of gradient evaluations is N+2
num_grad_eval = 0
for depth_index in range(burn_in, len(nuts_traj)):
    if both_directions[depth_index]:
        num_grad_eval += pow(2,nuts_traj[depth_index]) + 2
    else:
        num_grad_eval += pow(2,nuts_traj[depth_index]) + 1

print("ESS with NUTS:", ess_hnn)
print("ESS/(number of gradient evaluation) with NUTS:", ess_hnn/num_grad_eval)

if args.plot_samples:
    fig, ax = plt.subplots(figsize =(10, 7))
    markevery = max(int(req_samples_nuts/15000),1)
    ax.scatter(nuts_samples[::markevery,0], nuts_samples[::markevery,1], marker='+')
    plt.savefig('nuts.pdf')
