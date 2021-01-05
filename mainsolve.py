# import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

# import standard utils
import csv
import os
import time
import sys
import os
import json

sys.path.append("utils")
# local import from the subfolder "utils" -> all the code
from args import args   #command-line arg parser
from param import param #parameter class

from helpers import memory, create_directories    #some helpers functions


from dataset_creation import dataset_class   #dataset class
from dataloader import dataloader   #dataloader class

from BayesNN import SVGD_BayesNN #BayesianNN for SVGD method
from BayesNN import MCMC_BayesNN #BayesianNN for every MCMC methods
from pde_constraint import eikonal, anisotropic_eikonal #pde constraint

from SVGD import SVGD   #SVGD method
from HMC_MCMC import HMC_MCMC   #HMC method

from compute_error import compute_error #class for computing errors and uncertainty
from plotter import plot_result, plot_axis_example, plot_losses, plot_log_betas, plot_all_result, plot_log_prob #plot functions


# load the json file with all the parameter
with open(os.path.join("config",args.config)) as hpFile:
    hp = json.load(hpFile)

# create a param object with hp (param from json file) and args (command-line param)
par = param(hp, args)

# print all the selected parameters
print("---------------------------------------------------")
par.print_parameter()
print("--------------------------------------------")

# build the directories
path_result, path_plot, path_weights = create_directories(par)

# save parameters
par.save_parameter(path_result)


print("--------------------------------------------")
print("Bayesian PINN with ", par.method)
print("Solve the inverse problem of "+str(par.n_input)+ "D " + par.pde+" Eikonal")
print("Dataset used: ", par.experiment["dataset"])
print("---------------------------------------------------")


print("--------------------------------------------")
print("Dataset creation...")
# Datasets Creation
datasets_class = dataset_class(par)

# plot the exact data
datasets_class.plot(path_plot)

print("Dataset created")
print("--------------------------------------------")


print("--------------------------------------------")
print("Build the PDE constraint...")
# build the pde constraint class that implements the computation of pde residual for each collocation point
# Depending on the choice of isotropic or anisotropic, call the right child class of "pde_constraint"
if(par.pde == "isotropic"):
    pde_constr = eikonal(par)
elif(par.pde == "anisotropic"):
    pde_constr = anisotropic_eikonal(par, datasets_class.an_constraints)
else:
    raise Exception("No other pde implemented")
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Initialize the Bayesian PINN...")
# Initialize the correct Bayesian NN (SVGD_BayesNN for "SVGD" method, MCMC_BayesNN for every other MCMC-like method)
if(par.method == "SVGD"):
    bayes_nn = SVGD_BayesNN(par.param_method["n_samples"], par.sigmas,
                            par.n_input, par.architecture,
                            par.n_output_vel, par.param, pde_constr,
                            par.param["random_seed"])
else:
    bayes_nn = MCMC_BayesNN(par.sigmas,
                            par.n_input, par.architecture,
                            par.n_output_vel, par.param, pde_constr,
                            par.param["random_seed"],
                            par.param_method["M_HMC"])
print("done")
print("--------------------------------------------")



print("--------------------------------------------")
print("Building dataloader...")
# Build the dataloader for minibatch training (of just collocation points)
batch_size = par.experiment["batch_size"]
reshuffle_every_epoch = True
batch_loader  = dataloader(datasets_class, batch_size, reshuffle_every_epoch)
batch_loader, batch_loader_size = batch_loader.dataload_collocation()
print("Done")
print("--------------------------------------------")



print("--------------------------------------------")
print("Building ", par.method ," alg...")
# Build the method class
if(par.method == "SVGD"):
    # Initialize SVGD
    alg = SVGD(bayes_nn, batch_loader, datasets_class, par.param_method)
elif(par.method == "HMC"):
    # Initialize HMC
    alg = HMC_MCMC(bayes_nn, batch_loader, datasets_class,
                par.param_method, par.param["random_seed"])
else:
    raise Exception("Method not found")

print("Done")
print("--------------------------------------------")


print("--------------------------------------------")
print('Start training...')
tic = time.time()
# training with the selected method
rec_log_betaD, rec_log_betaR, LOSS,LOSS1,LOSS2,LOSSD = alg.train_all(par.utils["verbose"], par.utils["save_every_n_epochs"])
training_time = time.time() - tic
print('finished in ',training_time) # total time of training
print("--------------------------------------------")


print("--------------------------------------------")
print("Saving networks weights...")
# save networks after training
bayes_nn.save_networks(path_weights)
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Save losses...")
# save losses in csv files
np.savetxt(os.path.join(path_result,"Loss.csv"),LOSS)
np.savetxt(os.path.join(path_result,"LOSS1.csv"),LOSS1)
np.savetxt(os.path.join(path_result,"LOSS2.csv"),LOSS2)
np.savetxt(os.path.join(path_result,"LOSSD.csv"),LOSSD)
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Save log betass...")
# save log_betaD and log_betaR
rec_log_betaD = np.array(rec_log_betaD) # shape = ( (n_epochs*n_coll/batch_size), num_neural_networks )
rec_log_betaR = np.array(rec_log_betaR)
np.save(os.path.join(path_result,"log_betaD.npy"),rec_log_betaD)
np.save(os.path.join(path_result,"log_betaR.npy"),rec_log_betaR)
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Computing errors...")
# create the class to compute results and error
c_e = compute_error(par.n_output_vel, bayes_nn, datasets_class, path_result)
# compute errors and return mean and std for both outputs
at_NN, v_NN, at_std, v_std, errors = c_e.error()
print("done")

print("--------------------------------------------")
print("Plotting the results...")
plot_result(par.n_output_vel, at_NN, v_NN, at_std, v_std, datasets_class, path_plot)
if(par.dataset_type == "analytical" and par.n_input>1):
    plot_axis_example(par.n_output_vel, datasets_class, bayes_nn, path_plot)
print("done")
print("--------------------------------------------")

print("--------------------------------------------")
print("Plotting the losses")
# #################################### LOSS #####################################
plot_losses(LOSSD, LOSS1, LOSS2, LOSS, path_plot)
# #plt.show()
print("done")
print("--------------------------------------------")

print("--------------------------------------------")
print("Plotting log betas")
# #################################### LOSS #####################################
plot_log_betas(rec_log_betaD, rec_log_betaR, path_plot)
# #plt.show()
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Save log prob")
# #################################### LOSS #####################################
#breakpoint()
eikonal_logloss = np.array(bayes_nn.eikonal_logloss)
data_logloss = np.array(bayes_nn.data_logloss)
prior_logloss = np.array(bayes_nn.prior_logloss)

plot_log_prob(eikonal_logloss, data_logloss, prior_logloss, path_plot)

np.save(os.path.join(path_result, "eikonal_logloss.npy"),eikonal_logloss)
np.save(os.path.join(path_result, "data_logloss.npy"),data_logloss)
np.save(os.path.join(path_result, "prior_logloss.npy"),prior_logloss)
# #plt.show()
print("done")
print("--------------------------------------------")


print("--------------------------------------------")
print("Plot all the NNs")
if(par.n_input == 1 or par.dataset_type == "analytical"):
    if(par.n_input == 1):
        inputs,at,v = datasets_class.get_dom_data()
    else:
        inputs,at,v = datasets_class.get_axis_data()

    at_NN, v_NN = bayes_nn.predict(inputs)
    x = inputs[:,0]
    plot_all_result(x, at, v, at_NN, v_NN, datasets_class, par.n_input, par.n_output_vel, par.method, path_plot)
else:
    print("Unable to plot all the NNs in 1D up to now")

print("done")
print("--------------------------------------------")
