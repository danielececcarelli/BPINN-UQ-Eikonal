import numpy as np
import tensorflow as tf

import time
import math

# import the modules you need
from helpers import memory

## list update: update left list usign right list and a mult factor
def list_update(left, right, mult):
    for i in range(len(left)):
        left[i]= left[i]+(mult*right[i])
    return left

### Hamiltonian Monte Carlo
class HMC_MCMC:
    """
    Hamiltonian Monte Carlo (MCMC)
    """

    def __init__(self, bayes_nn,train_loader,datasets_class, hmc_param, random_seed):
        """!
        @param bayes_nn an object of type MCMC_BayesNN that collects the neural network and all the methods to compute the posterior
        @param train_loader the data loader in minibatch for the collocation points
        @param datasets_class
        @param parameters parameter for the HMC method (in json file, "HMC" section)
        @param random_seed random seed for numpy
        """

        ## BayesNN
        self.bayes_nn = bayes_nn
        ## Dataloader in batch of collocation data after shuffle
        self.train_loader = train_loader
        ## datasets_class, store all the datasets (collocation, exact, exact_with_noise etc.)
        self.datasets_class = datasets_class

        ## Total number of iterations
        self.N = hmc_param["N_HMC"]
        ## Number of iterations after burn-in period
        self.M = hmc_param["M_HMC"]
        ## Number of Leapfrog step
        self.L = hmc_param["L_HMC"]
        ## Step size for NN parameters
        self.dt = hmc_param["dt_HMC"]
        ## Step size for log betas (if trainable)
        self.dt_noise = hmc_param["dt_noise_HMC"]

        ## Set the random seed
        np.random.seed(random_seed)

    def u_fun(self, sp_inputs, sp_target, inputs):
        """!
        Compute the function U(theta) in HMC algorithm. Return u_theta and
        all the three components of the posterior and
        all the three losses.

        @param sp_inputs sparse exact input
        @param sp_target sparse exact (noisy) activation times
        @param collocation points input
        """
        ## compute our prediction on the sparse inputs, collecting our sparse activation times
        sp_output_times, _ = self.bayes_nn.forward(sp_inputs)

        ## compute the log likelihood of exact data (sp_target - sp_output_times) and log prior
        log_likelihood_total, loss_d_scalar, log_likelihood,log_prior_w = \
                self.bayes_nn.log_joint(sp_output_times, sp_target)

        ## compute log likelihood of pde equation and losses for batch collocation points (PDE constraints)
        log_eq, loss_1_scalar, loss_2_scalar =  self.bayes_nn.pde_logloss(inputs)

        ## compute u_theta
        log_total = log_likelihood_total + log_eq[0]
        u_theta = -log_total

        return u_theta, log_likelihood, log_prior_w, log_eq, loss_1_scalar, loss_2_scalar, loss_d_scalar

    def h_fun(self,u,r):
        """!
        Compute hamilotnian H(U,r) = U + 1/2*(r'*M*r)
        (Here we consider M = c*Identity, where c is set to 0.1 :
            H(U,r) = U + 1/20*(||r||^2))

        @param u U(theta)
        @param r vector r
        """
        ## compute the (norm of r)^2
        r_square = 0.
        for rr in r:
            r_square += np.sum(rr**2)

        ## H(U,r) = U + 1/2*(r'*M*r) = U + 1/20*(||r||^2)) since M=0.1*Identity
        return (u + (1./20.)*r_square)

    def alpha_fun(self,u,r,u0,r0, iter):
        """!
        Compute alpha=acceptance rate. We modify the formula in order to prevent getting stuck in local minimum.
        First compute h = H(U,r)-H(U0,r0).
        If this is positive -> we want to reject, so we return an high alpha (max 0.95 and not 1.0 for the previous reason)
        If this is negative we'll return a  small alpha, using the formula alpha = exp(2*h), with h negative

        @param u actual U
        @param r actual r
        @param u0 previous U (U0)
        @param r0 previous r (r0)
        """
        ## compute h
        h = self.h_fun(u,r)-self.h_fun(u0,r0)

        ## if h is positive we set to zero just to avoid problem of overfloating with exponential
        if h>0:
            h = 0.

        ## alpha_max goes from 0.95 to 1.0 with iteration
        alpha_max = 0.95 + 0.05*(iter/self.N)
        ## compute alpha as min between alpha_max ans exp(2h)
        alpha = min(alpha_max, math.exp(2.*h))
        return alpha


    @tf.function # decorator @tf.function to speed up the computation
    def grad_U_theta(self, sp_inputs, sp_target, inputs):
        """!
        Compute gradient of U wrt theta (param of NN and log betas if trainable)
        through backpropagation (using autodifferentiation -> tf.GradientTape)

        @param sp_inputs inputs of noisy exact data, shape=(n_exact, n_input)
        @param sp_target activation times of noisy exact data, shape=(n_exact, 1)
        @param inputs inputs of collocation data (batch), shape=(batch_size, n_input)
        """
        ## parameters of the NN
        param = self.bayes_nn.get_trainable_weights()
        ## flag if there are some log_betas (or scalar v) trainable
        flag = self.bayes_nn.log_betas.betas_trainable_flag()
        if(flag):
            ## if there are, collect them
            betas_trainable = self.bayes_nn.log_betas.get_trainable_log_betas()

        ## GradientTape
        with tf.GradientTape(persistent=True) as tape:
            ## watch all the parameters of the NN
            tape.watch(param)
            if(flag):
                ## if flag=True watch also log_betas trainable
                tape.watch(betas_trainable)
            ## Compute U(theta) calling the u_fun method
            u_theta, log_likelihood, log_prior_w, log_eq,*losses = self.u_fun(sp_inputs, sp_target, inputs)
        ## compute the gradient of NN param (by backpropagation)
        grad_theta = tape.gradient(u_theta, param)
        grad_theta = grad_theta[0]
        ## if flag=True compute also the gradient of every log_beta and append it to grad_theta
        if(flag):
            for log_beta in betas_trainable:
                grad_theta.append( tape.gradient(u_theta, log_beta) )
        ## delete the tape
        del tape
        return grad_theta, u_theta, log_likelihood, log_prior_w, log_eq, losses

    def train_all(self, verbosity, save_every_n_epochs): # TODO: add save_every_n_epochs!
        """ Train using HMC algorithm """

        loss_1 = 1.    # Initialize the losses (only in case we reject the first iteration)
        loss_2 = 1.
        loss_d = 1.

        rec_log_betaD = []  # list that collects all the log_betaD during training
        rec_log_betaR = []  # list that collects all the log_betaR during training
        LOSS = []   # list that collects total loss during training
        LOSS1 = []  # list that collects loss of pde during training
        LOSS2 = []  # list that collects loss of high gradients during training
        LOSSD = []  # list that collects loss of exact noisy data during training

        thetas = [] # list to collect all the parameters of NN during training
        log_betaDs = [] # list to collect all the log_betaD of NN during training (TODO : use rec_log_betaD?)
        log_betaRs = [] # list to collect all the log_betaR of NN during training

        ## Initialize all the "previous step" things we'll need
        theta0 = self.bayes_nn.nnets[0].get_weights().copy()
        len_theta = len(theta0)
        u_theta0 = 1e+9
        r0 = []

        ## To compute the % of accepted/iterations (acceptance rate, in all the training and only after burn-in)
        accepted_total = 0
        accepted_after_burnin = 0

        ## get noisy sparse exact data
        sp_inputs ,sp_at,sp_v = self.datasets_class.get_exact_data_with_noise()
        sp_target = sp_at

        ## for every iteration in 1,...,N
        for iteration in range(self.N):
            print(iteration, " iteration")

            epochtime = time.time()

            ## save the previous theta->theta0 (make a copy() of the list)
            theta0 = self.bayes_nn.nnets[0].get_weights().copy()
            ## collect theta
            theta = self.bayes_nn.nnets[0].get_weights()

            ## compute the auxiliary momentum variable rr
            ## rr will be a list of the same shape of theta and theta0
            rr = []
            ## for every i in len(theta0)
            for i in range(len(theta0)): # i = 2*hidden_layer + 2
                if i%2==0:
                    ## append a matrix of shape=(theta0[i].shape[0], theta0[i].shape[1]))
                    ## (W matrix of that layer) of Normal(0,1) values
                    rr.append(np.random.randn(theta0[i].shape[0], theta0[i].shape[1]))
                else:
                    ## append a vector of shape=(theta0[i].shape[0])
                    ## (b bias vector of that layer) of Normal(0,1) values
                    rr.append(np.random.randn(theta0[i].shape[0],))
            ## save the rr before we update it
            r0 = rr.copy()

            ## if we have additional trainable (log betas) create everything we need
            if(self.bayes_nn.log_betas.betas_trainable_flag()):
                rr_log_b = []   ## rr for log betas
                r0_log_b = []
                theta0_log_b = [] ## theta for log betas
                theta_log_b = []
                ## for every log b trainable we have
                for log_b in self.bayes_nn.log_betas.get_trainable_log_betas():
                    theta_log_b.append(log_b)   ## append the trainable log_b
                    rr_log_b.append(np.random.randn(1)) ## append a normal(0,1) values for every log_b
                theta0_log_b = theta_log_b.copy() ## theta0_log_b is just a copy() (previous)
                r0_log_b = rr_log_b.copy() ##r0_log_b is just a copy() (previous)

            ## for every step in 1,...,L (L = LEAPFROG STEPS)
            for s in range(self.L):
                ## iterate over all the batches of collocation data
                for batch_idx,inputs in enumerate(self.train_loader):

                    #############################################################
                    #################      Leapfrog step   ######################
                    #############################################################

                    ## backpropagation using method grad_U_theta
                    grad_theta, u_theta, log_likelihood, log_prior_w, log_eq, losses = \
                        self.grad_U_theta(sp_inputs, sp_target, inputs)

                    ## update rr = rr - (dt/2)*grad_theta
                    rr = list_update(rr, grad_theta, -self.dt/2)
                    ## if betas_trainable update also rr_log_b
                    if(self.bayes_nn.log_betas.betas_trainable_flag()):
                        rr_log_b = list_update(rr_log_b, grad_theta[len_theta:], -self.dt_noise/2)

                    ## update theta = theta + dt*rr
                    theta = list_update(theta, rr, self.dt)
                    ## update the weights in the NN with the new theta
                    self.bayes_nn.nnets[0].update_weights(theta)

                    ## if betas_trainable update also theta_log_b
                    if(self.bayes_nn.log_betas.betas_trainable_flag()):
                        theta_log_b = list_update(theta_log_b, rr_log_b, self.dt_noise)
                        self.bayes_nn.log_betas.log_betas_update(theta_log_b)

                    ## backpropagation using method grad_U_theta (now with the new theta)
                    grad_theta, u_theta, log_likelihood, log_prior_w, log_eq, losses = \
                        self.grad_U_theta(sp_inputs, sp_target, inputs)

                    ## update rr = rr - (dt/2)*grad_theta
                    rr = list_update(rr, grad_theta, -self.dt/2)
                    ## if betas_trainable update also rr_log_b
                    if(self.bayes_nn.log_betas.betas_trainable_flag()):
                        rr_log_b = list_update(rr_log_b, grad_theta[len_theta:], -self.dt_noise/2)

                    ## save all the three posterior components
                    self.bayes_nn.data_logloss.append(log_likelihood)
                    self.bayes_nn.prior_logloss.append(log_prior_w)
                    self.bayes_nn.eikonal_logloss.append(log_eq)



            debug_flag = True
            if(debug_flag and iteration>0):
                print("u_theta0: ",u_theta0.numpy())
                print("u_theta: ",u_theta.numpy())
                print("------------------------------")
                print("*******************************")
                print("Log likelihood: ")
                print(log_likelihood.numpy()[0][0])
                print("*******************************")
                print("Log prior w: ")
                print(log_prior_w.numpy()[0])
                print("*******************************")
                print("Log eq: ")
                print(log_eq.numpy()[0])
                print("-------------------------------")

            if(debug_flag and iteration>0 and self.bayes_nn.log_betas.betas_trainable_flag()):
                print(theta_log_b)

            if(verbosity):
                if(iteration == 0 or iteration==(self.N-1)):
                    mem = memory()
                    print("total memory used = ", mem)

                fin_epochtime = time.time()-epochtime
                print("time for this iteration = ", fin_epochtime)

            ## accept vs reject step
            ## sample p from a Uniform(0,1)
            p = np.random.random()
            ## compute alpha prob using alpha_fun (since now alpha is 0.95 at most,
            ## we can have some instabilities and ending up with a NaN, see after)
            alpha = self.alpha_fun(u_theta,rr,u_theta0,r0, iteration)

            if(debug_flag):
                print("alpha: ", alpha)
                print("p: ", p)

            ## if p>=alpha (and u_theta is not a NaN)
            ##          ACCEPT THE NEW VALUES
            if(p>=alpha and not math.isnan(u_theta)):
                ## update_weights with the new theta
                self.bayes_nn.nnets[0].update_weights(theta)
                ## if betas_trainable_flag=True update also the other trainable parameters
                if(self.bayes_nn.log_betas.betas_trainable_flag()):
                    self.bayes_nn.log_betas.log_betas_update(theta_log_b)
                    if(self.bayes_nn.log_betas._bool_log_betaD and self.bayes_nn.log_betas._bool_log_betaR):
                        log_betaDs.append(theta_log_b[0].numpy())
                        log_betaRs.append(theta_log_b[1].numpy())
                    else:
                        if(self.bayes_nn.log_betas._bool_log_betaD):
                            log_betaDs.append(theta_log_b[0].numpy())
                        else:
                            log_betaRs.append(theta_log_b[0].numpy())

                # store the new theta
                thetas.append(theta)
                print("accept")

                # update accepted_total and accepted_after_burnin
                accepted_total+=1
                if(iteration>(self.N-self.M)):
                    accepted_after_burnin+=1

                loss_1 = losses[0].numpy()
                loss_2 = losses[1].numpy()
                loss_d = losses[2].numpy()

                print("loss 1: ",loss_1)
                print("loss 2: ",loss_2)
                print("loss d: ",loss_d)

                # update theta0 and u_theta0
                theta0 = theta.copy()
                u_theta0 = u_theta


            else:
                ##          REJECT THE NEW VALUES
                ## update weights with the previous theta0
                self.bayes_nn.nnets[0].update_weights(theta0)
                ## if betas_trainable_flag=True update also the other trainable parameters (with the previous values)
                if(self.bayes_nn.log_betas.betas_trainable_flag()):
                    self.bayes_nn.log_betas.log_betas_update(theta0_log_b)
                    if(self.bayes_nn.log_betas._bool_log_betaD and self.bayes_nn.log_betas._bool_log_betaR):
                        log_betaDs.append(theta0_log_b[0].numpy())
                        log_betaRs.append(theta0_log_b[1].numpy())
                    else:
                        if(self.bayes_nn.log_betas._bool_log_betaD):
                            log_betaDs.append(theta0_log_b[0].numpy())
                        else:
                            log_betaRs.append(theta0_log_b[0].numpy())

                ## store the previous losses since we have rejected (after the first iteration where we don't have previous values )
                if(iteration>0):
                    loss_1 = LOSS1[-1]
                    loss_2 = LOSS2[-1]
                    loss_d = LOSSD[-1]

                ## store theta0
                thetas.append(theta0.copy())
                print("reject")

                ## Handling NaN values:
                ## Since we could end up with NaN (because we have changed alpha from max 1.0 to max 0.95)
                ## when we have a NaN (that usually happens when we have a very bad solution but our sampled p is >0.95),
                ## we recall a previous solutions (4 step behind for instance) that usually is more stable
                if(math.isnan(u_theta) and iteration>3):
                    self.bayes_nn.nnets[0].update_weights(thetas[-4])
                    if(self.bayes_nn.log_betas.betas_trainable_flag()):
                        if(self.bayes_nn.log_betas._bool_log_betaD):
                            self.bayes_nn.log_betas.log_betaD.assign(log_betaDs[-4])
                        if(self.bayes_nn.log_betas._bool_log_betaR):
                            self.bayes_nn.log_betas.log_betaR.assign(log_betaRs[-4])
            ## store all the losses and log_betas
            LOSS1.append(loss_1)
            LOSS2.append(loss_2)
            LOSSD.append(loss_d)
            LOSS.append(loss_1+loss_2+loss_d)
            rec_log_betaD.append(self.bayes_nn.log_betas.log_betaD.numpy())
            rec_log_betaR.append(self.bayes_nn.log_betas.log_betaR.numpy())

        ## print accepance rates
        print("total accepance rate: ", accepted_total/self.N)
        print("after_burn_in accepance rate: ", accepted_after_burnin/self.M)

        ## store thetas and log_betas(if trainable) (just the last M iterations) in bayes_nn
        ## so we can compute all the statistics we need
        self.bayes_nn._thetas = thetas[-self.M:]
        if(self.bayes_nn.log_betas._bool_log_betaD):
            self.bayes_nn._log_betaDs = log_betaDs[-self.M:]
        if(self.bayes_nn.log_betas._bool_log_betaD):
            self.bayes_nn._log_betaRs = log_betaRs[-self.M:]

        return rec_log_betaD, rec_log_betaR, LOSS,LOSS1,LOSS2,LOSSD
