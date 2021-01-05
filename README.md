# BPINN for Inverse Eikonal
## Bayesian Physics-Informed Neural Networks for Uncertainty Quantification in Inverse Eikonal problem

## Intro

A Bayesian implementation of Physics-Informed Neural Network to solve the Inverse Problem (identification of the fields of conduction velocity tensor) in Eikonal (isotropic or anisotropic) equation. Thanks to the bayesian framework we can also use noisy data as input and estimate the Uncertainty of our result (Uncertainty Quantification problem).

## Run

How to run the code:

### 1) choose a method for Bayesian PINN

Methods available:

- HMC (Hamiltonian Monte Carlo), one the classical MCMC algorithm to sample from posterior
- SVGD (Stein Variational Gradient Descent), a finite method that approximate the posterior through N finite particles


### 2) run

You have to specify at least (mandatory!) the method you want to use.

(run on CPU):

- CUDA_VISIBLE_DEVICES="" python mainsolve.py --method HMC
- CUDA_VISIBLE_DEVICES="" python mainsolve.py --method SVGD


(run on GPU):

- python mainsolve.py --method HMC
- python mainsolve.py --method SVGD

### 3) parameters

You can specify in a json file all the parameters you are going to use in the code. You can find some example in "config" subfolder (for instance all the experiment reported in the report.pdf) and a "default.json", that is the default choice if you don't provide any config file. If you want to specify your own json file you can:
- make a copy of the "default.json", rename it as "yourname.json" and put it in the "config" subfolder
- modify your "yourname.json" in an appropriate way (you will have a dictionary for both "HMC" and "SVGD" parameters inside your file, you can delete the one you are not going to use if you want...)
- run with the additional arguments --config yourname.json, for instance:

- python mainsolve.py --method SVGD --config yourname.json

In "utils/args.py" you can find also an arg parser for additional parameters provided by command-line. You can overspecify some parameters directly from command-line without the need of create a new json file for just a small change. This can be useful when you want to test the algorithm with a fixed set of parameters and varying just one, for instance:

- python mainsolve.py --method SVGD --config default.json (where n_samples is set to 30 in default.json file)
- python mainsolve.py --method SVGD --config default.json --n_samples 20 (now we are setting n_samples to 20)
- python mainsolve.py --method SVGD --config default.json --n_samples 40 (now we are setting n_samples to 40)

### 4) config, dataset and experiment available

The parameter you can specify in a json file are:
{
"architecture":{"n_layers":5, "n_neurons":100},

"experiment":{"dataset":"exponential", "prop_exact":0.01, "prop_collocation":1.00, "is_uniform_exact":"False", "noise_lv":0.01, "batch_size":100},

"param":{	"param_eikonal":0.1, "param_log_joint":2.0,	"param_prior":1.0,	"param2loss":1e-4,	"random_seed":42	},

"sigmas":{	"data_prior_noise":1e-4,	"pde_prior_noise":1e-4,	"data_prior_noise_trainable":"False",	"pde_prior_noise_trainable":"False"	},

"SVGD":{	"n_samples":30,	"epochs":100,	"lr":1e-3,	"lr_noise":1e-5,	"param_repulsivity":100.0	},

"HMC":{	"N_HMC":1000,	"M_HMC":750,	"L_HMC":10,	"dt_HMC":1e-3	},

"utils":{	"verbose":"True",	"save_flag":"True",	"save_every_n_epochs":0	}
}

In architecture you can specify the structure of the Feed Forward Neural Network (num of layer and num of neurons for each layer)
In experiment you can choose which dataset you want to use, how many points do you want to use as exact data and collocation data etc.
In param you can specify some weights for the Bayesian computation of the posterior (and a param for penalizing high gradients of velocity)
In sigmas you can provide additional parameter if you want to make sigmas trainable hyperparameter
In "SVGD" and "HMC" you can specify the parameters for the selected method
In "utils" some other flag for verbosity and saving result

You can select some different datasets, some of them built using the analytical solution of the Eikonal equation, some other collected using Pykonal library, a python library to solve the Forward Eikonal isotropic problem.

You can choose between:
- "exponential": 1D, isotropic, analytical
- "circle": 2D, isotropic, analytical
- "triflag": 2D, isotropic, pykonal_dataset
- "square_with_circle": 2D, isotropic, pykonal_dataset
- "anisotropic1": 2D, anisotropic, analytical
- "anisotropic2": 2D, anisotropic, analytical
- "prova3D": 3D, isotropic, pykonal_dataset
- "prolate3D": 3D, isotropic, pykonal_dataset
- "prolate3D_scar_new_version": 3D, isotropic, pykonal_dataset
- "prolate3D_new_version": 3D isotropic, pykonal_dataset

You will find all the result in the correct subfolder:
- 1D-isotropic-eikonal
- 2D-anisotropic-eikonal
- 2D-isotropic-eikonal
- 3D-isotropic-eikonal

## DOCS
All the docs (generated with DOxygen) can be found here [readme_docs](https://danielececcarelli.github.io/BPINN-UQ-Eikonal/html/index.html)
