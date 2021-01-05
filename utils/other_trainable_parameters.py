import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class trainable_param:
    """
    A class to store all the additional trainable parameters, for instance log_betaD and log_betaR
    where log_betaD = log(betaD) = log(1 / sigmaD^2) and
          log_betaR = log(betaR) = log(1 / sigmaR^2)
    """
    def __init__(self, bool_log_betaD, bool_log_betaR, param, num_neural_networks, random_seed,
                bool_v_scalar=False, v_scalar_initial_mean=1., v_scalar_initial_std = 0.1):
        """! Constructor of trainable_param

        @param bool_log_betaD flag for trainability of log_betaD
        @param bool_log_betaR flag for trainability of log_betaR
        @param param parameters to compute log_betaD and log_betaR
        @param num_neural_networks 1 if HMC, num_neural_networks for SVGD
        @param random_seed random_seed for tensorflow_probability sampling
        @param bool_v_scalar (optional) if we are in experiment v_scalar
        @param v_scalar_initial_mean (optional)
        @param v_scalar_initial_std (optional)
        """
        ## flag for trainability of log_betaD
        self._bool_log_betaD = bool_log_betaD #for sigmaD
        ## flag for trainability of log_betaR
        self._bool_log_betaR = bool_log_betaR #for sigmaR
        ## flag for trainability of v_scalar
        self._bool_v_scalar = bool_v_scalar #only used in v_scalar UQ experiment

        tf.random.set_seed(random_seed)


        if param != None:
            ## if we provide actual parameters
            beta_prior_shape = param["beta_prior_shape"]
            beta_prior_rate = param["beta_prior_rate"]
            beta_pde_shape = param["beta_pde_shape"]
            beta_pde_rate = param["beta_pde_rate"]

            ## sample from a log (Gamma distribution)
            log_betaD = tf.dtypes.cast(tf.math.log(tfd.Gamma(beta_prior_shape,
                        beta_prior_rate).sample(sample_shape=(num_neural_networks,), seed=random_seed)), dtype=tf.float64)
            log_betaR = tf.dtypes.cast(tf.math.log(tfd.Gamma(beta_pde_shape,
                        beta_pde_rate).sample(sample_shape=(num_neural_networks,), seed=random_seed )), dtype=tf.float64)

        else:
            ## else put them equal to an initial values 9.0
            log_betaD = tf.ones(shape=(num_neural_networks,), dtype=tf.float64)
            log_betaR = tf.ones(shape=(num_neural_networks,), dtype=tf.float64)
            log_betaD*=9.0
            log_betaR*=9.0

        ## log_betaD tensorflow variable
        self.log_betaD = tf.Variable(log_betaD, trainable = self._bool_log_betaD, dtype=tf.float64, name="log_betaD")
        ## log_betaR tensorflow variable
        self.log_betaR = tf.Variable(log_betaR, trainable = self._bool_log_betaR, dtype=tf.float64, name="log_betaR")

        if(bool_v_scalar): #otherwise we dont need it
            v_scalar = tfd.Normal(loc=v_scalar_initial_mean, scale=v_scalar_initial_std).sample(sample_shape=(num_neural_networks,))
            self.v_scalar = tf.Variable(v_scalar, trainable = True, dtype=tf.float64, name="v_scalar")

    def get_trainable_log_betas(self):
        """ return a list of all the trainable variable we have in our attributes
        """
        list_betas = []
        if(self._bool_log_betaD):
            list_betas.append(self.log_betaD)
        if(self._bool_log_betaR):
            list_betas.append(self.log_betaR)
        if(self._bool_v_scalar):
            list_betas.append(self.v_scalar)
        return list_betas

    def betas_trainable_flag(self):
        """Return True if we have at least one trainable variable"""
        return (self._bool_log_betaD or self._bool_log_betaR or self._bool_v_scalar)

    def log_betas_update(self, theta):
        """Update log betas proving a list of values (len = 2 if both are trainable, 1 otherwise)"""
        if(self.betas_trainable_flag()):

            # if both are trainable
            if(self._bool_log_betaD and self._bool_log_betaR):
                self.log_betaD.assign(theta[0])
                self.log_betaR.assign(theta[1])
            # otherwise
            else:
                if(self._bool_log_betaD):
                    self.log_betaD.assign(theta[0])
                else:
                    self.log_betaR.assign(theta[0])
