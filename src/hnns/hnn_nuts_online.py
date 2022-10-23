# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Parts of this code were borrowed from https://github.com/mfouesneau/NUTS which has an MIT License
# No-U-Turn Sampling with HNNs

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
from data import dynamics_fn

##### User-defined sampling parameters #####

N = 1000 # number of samples
burn = 100 # number of burn-in samples
epsilon = 0.025 # step size
N_lf = 20 # number of cool-down samples when HNN integration errors are high (see https://arxiv.org/abs/2208.06120)
hnn_threshold = 10. # HNN integration error threshold (see https://arxiv.org/abs/2208.06120)
lf_threshold = 1000. # Numerical gradient integration error threshold

##### Sampling code below #####

y0 = np.zeros(args.input_dim)
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

hnn_model = get_model(args, baseline=False)

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)

# def logsumexp1(a, b):
#     c = log(np.exp(a)+np.exp(b))
#     return c

def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf):
    """The main recursion."""
    if (j == 0):
        # joint0 = hamil(hnn_ivp1[:,1])
        t_span1 = [0,v * epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        y1 = np.concatenate((theta, r), axis=0)
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
        thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
        rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
        joint = functions(hnn_ivp1[:,1])
        # nprime = int(logu <= np.exp(-joint)) # int(logu <= (-joint)) #  int(logu < joint) # 
        call_lf = call_lf or int((np.log(logu) + joint) > 10.) # int(logu <= np.exp(10. - joint)) # int((logu - 10.) < joint) # int((logu - 10.) < joint) #  int(tmp11 <= np.minimum(1,np.exp(joint0 - joint))) and int((logu - 1000.) < joint) 
        monitor = np.log(logu) + joint # sprime
        sprime = int((np.log(logu) + joint) <= hnn_threshold) # 
        
        if call_lf:
            t_span1 = [0,v * epsilon]
            y1 = np.concatenate((theta, r), axis=0)
            hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y1, 1, int(args.input_dim))
            thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
            rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
            joint = functions(hnn_ivp1[:,1])
            sprime = int((np.log(logu) + joint) <= lf_threshold)
        
        nprime = int(logu <= np.exp(-joint))
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        alphaprime = min(1., np.exp(joint0 - joint))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree(theta, r, logu, v, j - 1, epsilon, joint0, call_lf)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf)
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                rprime = rprime2[:]
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

D = int(args.input_dim/2)
M = N
Madapt = 0
theta0 = np.ones(D)
D = len(theta0)
samples = np.empty((M + Madapt, D), dtype=float)
samples[0, :] = theta0
y0 = np.zeros(args.input_dim)
for ii in np.arange(0,int(args.input_dim/2),1):
    y0[ii] = norm(loc=0,scale=1).rvs()
for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    y0[ii] = norm(loc=0,scale=1).rvs()
HNN_accept = np.ones(M)
traj_len = np.zeros(M)
alpha_req = np.zeros(M)
H_store = np.zeros(M)
monitor_err = np.zeros(M)
call_lf = 0
counter_lf = 0
is_lf = np.zeros(M)

for m in np.arange(1, M + Madapt, 1):
    print(m)
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #
    
    joint = functions(y0) # logp - 0.5 * np.dot(r0, r0.T)

    logu = np.random.uniform(0, np.exp(-joint))

    samples[m, :] = samples[m - 1, :]

    # initialize the tree
    thetaminus = samples[m - 1, :]
    thetaplus = samples[m - 1, :]
    rminus = y0[int(args.input_dim/2):int(args.input_dim)]
    rplus = y0[int(args.input_dim/2):int(args.input_dim)]
    
    j = 0  # initial heigth j = 0
    n = 1  # Initially the only valid point is the initial point.
    s = 1  # Main loop: will keep going until s == 0.
    # call_lf = 0
    if call_lf:
        counter_lf +=1
    if counter_lf == N_lf:
        call_lf = 0
        counter_lf = 0

    while (s == 1):
        # Choose a direction. -1 = backwards, 1 = forwards.
        v = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if (v == -1):
            thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint, call_lf)
        else:
            _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint, call_lf)

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        _tmp = min(1, float(nprime) / float(n))
        if (sprime == 1) and (np.random.uniform() < _tmp):
            samples[m, :] = thetaprime[:]
            r_sto = rprime
        # Update number of valid points we've seen.
        n += nprime
        # Decide if it's time to stop.
        s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # Increment depth.
        j += 1
        monitor_err[m] = monitor
        
    is_lf[m] = call_lf
    traj_len[m] = j
    alpha_req[m] = alpha
    y0[0:int(args.input_dim/2)] = samples[m, :]
    H_store[m] = functions(np.concatenate((samples[m, :], r_sto), axis=0))
    # alpha1 = 1.
    # if alpha1 > uniform().rvs():
        
    # else:
    #     samples[m, :] = samples[m-1, :]
    #     HNN_accept[m] = 0
    #     H_store[m] = joint
    
hnn_tf = tf.convert_to_tensor(samples[burn:M,:])
ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))