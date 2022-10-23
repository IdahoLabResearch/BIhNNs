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
import tensorflow_probability as tfp
from get_args import get_args
args = get_args()

## Load data and train SympNet (LA or G)

path = '{}/{}.pkl'.format(args.load_dir, args.load_file_name)
data_raw = from_pickle(path)
print("Successfully loaded data")
d1 = data_raw['coords']
xt = np.zeros((int(args.len_sample**2 * args.num_samples), args.input_dim))
yt = np.zeros((int(args.len_sample**2 * args.num_samples), args.input_dim))
rn = np.arange(0,args.num_samples+1) + np.arange(0,int(args.len_sample**2 * args.num_samples)+1,int(args.len_sample**2))
for ii in range(args.num_samples):
    for jj in range(int(args.input_dim/2)):
        xt[int(args.len_sample**2)*ii:int(args.len_sample**2)*(ii+1),int(args.input_dim/2)+jj] = d1[rn[ii]:rn[ii+1]-1,jj]
        yt[int(args.len_sample**2)*ii:int(args.len_sample**2)*(ii+1),int(args.input_dim/2)+jj] = d1[rn[ii]+1:rn[ii+1],jj]
    for jj in range(int(args.input_dim/2)):
        xt[int(args.len_sample**2)*ii:int(args.len_sample**2)*(ii+1),jj] = d1[rn[ii]:rn[ii+1]-1,int(args.input_dim/2)+jj]
        yt[int(args.len_sample**2)*ii:int(args.len_sample**2)*(ii+1),jj] = d1[rn[ii]+1:rn[ii+1],int(args.input_dim/2)+jj]

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
ln.Brain.Run()
ln.Brain.Restore()
ln.Brain.Output()

## Sampling parameters

req_samples = 25000
hmc_len = 500
burn_in = 1000

## Sampling

# Langevin Monte Carlo
samples, lmc_accept = LMC(net,req_samples)
# Hamiltonian Monte Carlo
samples, hmc_accept = HMC(net,req_samples,hmc_len = 500)
# No-U-Turn Sampling with online error monitoring
samples, nuts_err, nuts_ind, nuts_traj = NUTS(net,req_samples)

## Compute effective sample size
hnn_tf = tf.convert_to_tensor(samples[burn_in:req_samples,:])
ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
