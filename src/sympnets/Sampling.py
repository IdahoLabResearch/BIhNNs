# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Parts of this code were borrowed from https://github.com/mfouesneau/NUTS which has an MIT License
# Langevin Monte Carlo, Hamiltonian Monte Carlo, and NUTS with online error monitoring with SympNets

import numpy as np
from functions import func1
from scipy.stats import norm
from scipy.stats import uniform
import torch
import autograd
from tqdm import tqdm
from get_args import get_args

args = get_args()
input_dim1 = int(args.input_dim/2)
chains = 1
y0 = np.zeros(int(input_dim1*2))
burn = 0
N_lf = 20 # number of cool-down samples when sympnet integration errors are high (see https://arxiv.org/abs/2208.06120)
hnn_threshold = 10. # sympnet integration error threshold (see https://arxiv.org/abs/2208.06120)
lf_threshold = 1000. # Numerical gradient integration error threshold

def LMC(net, N):
    y0 = np.zeros(int(input_dim1*2))
    steps = 1
    hnn_fin = np.zeros((chains,N,int(int(input_dim1*2)/2)))
    hnn_accept = np.zeros((chains,N))
    for ss in np.arange(0,chains,1):
        x_req = np.zeros((N,int(int(input_dim1*2)/2)))
        x_req[0,:] = y0[0:int(int(input_dim1*2)/2)]
        accept = np.zeros(N)

        for ii in np.arange(0,int(int(input_dim1*2)/2),1):
            y0[ii] = 0.0
        for ii in np.arange(int(int(input_dim1*2)/2),int(int(input_dim1*2)),1):
            y0[ii] = norm(loc=0,scale=1).rvs()
        HNN_sto = np.zeros((int(input_dim1*2),1,N))
        for ii in tqdm(range(N), desc="Sampling using LMC, chain "+str(ss)):
            y0 = np.concatenate((y0[int(int(input_dim1*2)/2):int(input_dim1*2)], y0[0:int(int(input_dim1*2)/2)]))
            hnn_ivp = net.predict(torch.tensor(y0.astype('float32')), steps, keepinitx=False, returnnp=True) # integrate_model(hnn_model, dt, y0)
            hnn_ivp = np.append(hnn_ivp[steps-1,int(int(input_dim1*2)/2):int(input_dim1*2)].squeeze(), hnn_ivp[steps-1,0:int(int(input_dim1*2)/2)].squeeze())
            for sss in range(0,int(input_dim1*2)):
                HNN_sto[sss,:,ii] = hnn_ivp[sss]
            yhamil = np.zeros(int(input_dim1*2))
            for jj in np.arange(0,int(input_dim1*2),1):
                yhamil[jj] = hnn_ivp[jj]
            H_star = func1(yhamil)
            H_prev = func1(y0)
            alpha = np.minimum(1,np.exp(H_prev - H_star))
            if alpha > uniform().rvs():
                y0[0:int(int(input_dim1*2)/2)] = hnn_ivp[0:int(int(input_dim1*2)/2)]
                x_req[ii,:] = hnn_ivp[0:int(int(input_dim1*2)/2)]
                accept[ii] = 1
            else:
                x_req[ii,:] = y0[0:int(int(input_dim1*2)/2)]
            for jj in np.arange(int(int(input_dim1*2)/2),int(input_dim1*2),1):
                y0[jj] = norm(loc=0,scale=1).rvs()
        hnn_accept[ss,:] = accept
        hnn_fin[ss,:,:] = x_req
    return hnn_fin.squeeze(), hnn_accept

def HMC(net, N, steps):
    y0 = np.zeros(int(input_dim1*2))
    hnn_fin = np.zeros((chains,N,int(int(input_dim1*2)/2)))
    hnn_accept = np.zeros((chains,N))
    for ss in np.arange(0,chains,1):
        x_req = np.zeros((N,int(int(input_dim1*2)/2)))
        x_req[0,:] = y0[0:int(int(input_dim1*2)/2)]
        accept = np.zeros(N)

        for ii in np.arange(0,int(int(input_dim1*2)/2),1):
            y0[ii] = 0.0
        for ii in np.arange(int(int(input_dim1*2)/2),int(int(input_dim1*2)),1):
            y0[ii] = norm(loc=0,scale=1).rvs()
        HNN_sto = np.zeros((int(input_dim1*2),1,N))
        for ii in tqdm(range(N), desc="Sampling using HMC, chain "+str(ss)):
            y0 = np.concatenate((y0[int(int(input_dim1*2)/2):int(input_dim1*2)], y0[0:int(int(input_dim1*2)/2)]))
            hnn_ivp = net.predict(torch.tensor(y0.astype('float32')), steps, keepinitx=False, returnnp=True) # integrate_model(hnn_model, dt, y0)
            hnn_ivp = np.append(hnn_ivp[steps-1,int(int(input_dim1*2)/2):int(input_dim1*2)].squeeze(), hnn_ivp[steps-1,0:int(int(input_dim1*2)/2)].squeeze())
            for sss in range(0,int(input_dim1*2)):
                HNN_sto[sss,:,ii] = hnn_ivp[sss]
            yhamil = np.zeros(int(input_dim1*2))
            for jj in np.arange(0,int(input_dim1*2),1):
                yhamil[jj] = hnn_ivp[jj]
            H_star = func1(yhamil)
            H_prev = func1(y0)
            alpha = np.minimum(1,np.exp(H_prev - H_star))
            if alpha > uniform().rvs():
                y0[0:int(int(input_dim1*2)/2)] = hnn_ivp[0:int(int(input_dim1*2)/2)]
                x_req[ii,:] = hnn_ivp[0:int(int(input_dim1*2)/2)]
                accept[ii] = 1
            else:
                x_req[ii,:] = y0[0:int(int(input_dim1*2)/2)]
            for jj in np.arange(int(int(input_dim1*2)/2),int(input_dim1*2),1):
                y0[jj] = norm(loc=0,scale=1).rvs()
        hnn_accept[ss,:] = accept
        hnn_fin[ss,:,:] = x_req
    return hnn_fin.squeeze(), hnn_accept

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)

def dynamics_fn(t, coords):
    # print("Here")
    dcoords = autograd.grad(func1)(coords) #
    # dcoords = getgrad(coords)
    dic1 = np.split(dcoords,2*input_dim1)
    S = np.concatenate([dic1[input_dim1]])
    for ii in np.arange(input_dim1+1,2*input_dim1,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,input_dim1,1):
        S = np.concatenate([S, -dic1[ii]])
    return S

def leapfrog ( dydt, tspan, y0, n, dim ):
  t0 = tspan[0]
  tstop = tspan[1]
  dt = ( tstop - t0 ) / n

  t = np.zeros ( n + 1 )
  y = np.zeros ( [dim, n + 1] )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ):
      t[0]   = t0
      for j in range ( 0, dim ):
          y[j,0] = y0[j]
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])
    else:
      t[i]   = t[i-1] + dt
      aold   = anew
      for j in range ( 0, int(dim/2) ):
          y[j,i] = y[j,i-1] + dt * ( y[(j+int(dim/2)),i-1] + 0.5 * dt * aold[(j+int(dim/2))] )
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])
      for j in range ( 0, int(dim/2) ):
          y[(j+int(dim/2)),i] = y[(j+int(dim/2)),i-1] + 0.5 * dt * ( aold[(j+int(dim/2))] + anew[(j+int(dim/2))] )
  return y #t,

def build_tree(net, theta, r, logu, v, j, epsilon, joint0, call_lf):
    """The main recursion."""
    if (j == 0):
        # joint0 = hamil(hnn_ivp1[:,1])
        t_span1 = [0,v * epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        y1 = np.concatenate((r, theta), axis=0)
        hnn_ivp1 = net.predict(torch.tensor(y1.astype('float32')), 1, keepinitx=False, returnnp=True) # integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
        thetaprime = hnn_ivp1[0, input_dim1:int(input_dim1*2)].reshape(input_dim1)
        rprime = hnn_ivp1[0, 0:input_dim1].reshape(input_dim1)
        tmp11 = np.append(hnn_ivp1[0,int(int(input_dim1*2)/2):int(input_dim1*2)].squeeze(), hnn_ivp1[0,0:int(int(input_dim1*2)/2)].squeeze())
        joint = func1(tmp11)
        # nprime = int(logu <= np.exp(-joint)) # int(logu <= (-joint)) #  int(logu < joint) #
        call_lf = call_lf or int((np.log(logu) + joint) > hnn_threshold) # int(logu <= np.exp(10. - joint)) # int((logu - 10.) < joint) # int((logu - 10.) < joint) #  int(tmp11 <= np.minimum(1,np.exp(joint0 - joint))) and int((logu - 1000.) < joint)
        monitor = np.log(logu) + joint # sprime
        sprime = int((np.log(logu) + joint) <= hnn_threshold) #

        if call_lf:
            t_span1 = [0,v * epsilon]
            y1 = np.concatenate((theta, r), axis=0)
            hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y1, 1, int(int(input_dim1*2))) # integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
            thetaprime = hnn_ivp1[0:int(int(input_dim1*2)/2), 1].reshape(int(int(input_dim1*2)/2))
            rprime = hnn_ivp1[int(int(input_dim1*2)/2):int(int(input_dim1*2)), 1].reshape(int(int(input_dim1*2)/2))
            joint = func1(hnn_ivp1[:,1])
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
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree(net, theta, r, logu, v, j - 1, epsilon, joint0, call_lf)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(net, thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(net, thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf)
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

def NUTS(net, N):
    D = int(int(input_dim1*2)/2)
    theta0 = np.ones(D)
    D = len(theta0)
    samples = np.empty((N, D), dtype=float)
    samples[0, :] = theta0
    monitor_err = np.zeros(N)
    call_lf = 0
    counter_lf = 0
    # N_lf = 20
    is_lf = np.zeros(N)
    HNN_accept = np.ones(N)
    traj_len = np.zeros(N)
    alpha_req = np.zeros(N)
    H_store = np.zeros(N)
    epsilon = 0.025
    for m in tqdm(range(N), desc="Sampling using NUTS"):
        for ii in np.arange(int(int(input_dim1*2)/2),int(int(input_dim1*2)),1):
            y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #
        # Resample momenta.
        # r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = func1(y0) # logp - 0.5 * np.dot(r0, r0.T)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        # logu = float(-joint - np.random.exponential(1, size=1)) # compute_slice(joint)
        logu = np.random.uniform(0, np.exp(-joint))

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        # lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = y0[int(int(input_dim1*2)/2):int(int(input_dim1*2))]
        rplus = y0[int(int(input_dim1*2)/2):int(int(input_dim1*2))]
        # gradminus = grad[:]
        # gradplus = grad[:]

        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.
        # call_lf = 0
        if call_lf:
            counter_lf +=1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0

        r_sto = np.zeros(int(int(input_dim1*2)/2))
        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(net, thetaminus, rminus, logu, v, j, epsilon, joint, call_lf)
            else:
                _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(net, thetaplus, rplus, logu, v, j, epsilon, joint, call_lf)

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
        y0[0:int(int(input_dim1*2)/2)] = samples[m, :]
        H_store[m] = func1(np.concatenate((samples[m, :], r_sto), axis=0))

    return samples, monitor_err, is_lf, traj_len
