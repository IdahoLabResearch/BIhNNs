Copyright (c) 2022 Battelle Energy Alliance, LLC

Licensed under MIT License, please see LICENSE for details

https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# BIhNNs

## BIhNNs: Bayesian Inference with (Hamiltonian and other) Neural Networks

* Train neural network architectures like deep neural nets (DNN), Neural ODEs, Hamiltonian neural nets (HNNs), and symplectic neural nets to learn probability distribution spaces.
* Use the trained neural net to perform sampling without requiring gradient information of the target probability density.
* State-of-the-art sampling schemes like Langevin Monte Carlo, Hamiltonian Monte Carlo, and No-U-Turn Sampling are available for use with the above-mentioned trained neural nets.

# Publications

The code in this repository is part of the following two papers available on arXiv:

* Dhulipala et al. (2022) Bayesian Inference with Latent Hamiltonian Neural Networks. https://arxiv.org/abs/2208.06120.
* Dhulipala et al. (2022) Physics-Informed Machine Learning of Dynamical Systems for Efficient Bayesian Inference. https://arxiv.org/abs/2209.09349.

The below figure presents the workflow for performing sampling with (Hamiltonian and other) neural networks.

![Figure](Schematic.png)

# Using the code

## Deep neural nets (DNNs) [literature: https://arxiv.org/abs/1906.01563]

* go to src/dnns/
* Include the Hamiltonian of the required probability distribution in the `functions.py` file. Some example probability distributions are already included. For information on the Hamiltonian of a probability distribution, see https://arxiv.org/pdf/1206.1901.pdf%20http://arxiv.org/abs/1206.1901.pdf.
* Adjust the parameters in `get_args.py`
* Run `train_dnn.py` to train the DNN model. The training data will be stored in a pkl file with the name the user specified in `get_args.py`. The trained DNN will be stored in a tar file with the name the user specified in `get_args.py`.
* Then run, either `dnn_lmc.py`, `dnn_hmc.py`, `dnn_nuts_online.py` to, respectively, perform Langevin Monte Carlo, Hamiltonian Monte Carlo, and No-U-Turn Sampling with the trained DNN. Note that the user specified sampling parameters can be adjusted in these files.
* For No-U-Turn Sampling, an online error monitoring scheme as described in (https://arxiv.org/abs/2208.06120) is used. To turn this feature off, set the `hnn_threshold` parameter in `dnn_nuts_online.py` to a large value like 1000.

## Hamiltonian neural nets (HNNs) [literature: https://arxiv.org/abs/1906.01563]

* go to src/hnns/
* Include the Hamiltonian of the required probability distribution in the `functions.py` file. Some example probability distributions are already included. For information on the Hamiltonian of a probability distribution, see https://arxiv.org/pdf/1206.1901.pdf%20http://arxiv.org/abs/1206.1901.pdf.
* Adjust the parameters in `get_args.py`
* Run `train_hnn.py` to train the HNN model. The training data will be stored in a pkl file with the name the user specified in `get_args.py`. The trained HNN will be stored in a tar file with the name the user specified in `get_args.py`.
* Then run, either `hnn_lmc.py`, `hnn_hmc.py`, `hnn_nuts_online.py` to, respectively, perform Langevin Monte Carlo, Hamiltonian Monte Carlo, and No-U-Turn Sampling with the trained HNN. Note that the user specified sampling parameters can be adjusted in these files.
* For No-U-Turn Sampling, an online error monitoring scheme as described in (https://arxiv.org/abs/2208.06120) is used. To turn this feature off, set the `hnn_threshold` parameter in `dnn_nuts_online.py` to a large value like 1000.

## Symplectic neural nets (sympnets) [literature: https://arxiv.org/abs/2001.03750]

* go to src/sympnets/
* Prerequisite: download the `learner` directory from https://github.com/jpzxshi/sympnets into the src/sympnets/ folder
* Training data generation: generate the pkl file generated from either src/dnns/ or src/hnns/ as described under DNNs or HNNs. Copy this pkl file to src/sympnets/ folder.
* Include the Hamiltonian of the required probability distribution in the `functions.py` file. Some example probability distributions are already included. For information on the Hamiltonian of a probability distribution, see https://arxiv.org/pdf/1206.1901.pdf%20http://arxiv.org/abs/1206.1901.pdf.
* Adjust the parameters in `get_args.py`
* In `main.py`, run the "Load data and train SympNet (LA or G)" portion of the code to load the training data and train an LA or G sympnet based.
* In `main.py`, adjust the options in "Sampling parameters" portion of the code
* In `main.py`, run "Sampling" portion of the code to perform Langevin Monte Carlo, Hamiltonian Monte Carlo, or No-U-Turn Sampling
* For No-U-Turn Sampling, an online error monitoring scheme as described in (https://arxiv.org/abs/2208.06120) is used. To turn this feature off, set the `hnn_threshold` parameter in `Sampling.py` to a large value like 1000.

## Neural ODES

Coming soon!!

# Author information

Som L. Dhulipala 

Computational Scientist in Uncertainty Quantification

Computational Mechanics and Materials department

Email: Som.Dhulipala@inl.gov 

Idaho National Laboratory

# Acknowledgements

The authors of the following open-source codes are thanked whose work is helpful to the **BIhNNs** repository:

* https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
* https://github.com/mfouesneau/NUTS under the MIT License
* https://github.com/jpzxshi/sympnets (no license information is provided)
