import numpy as np

##############################################################################
########################         exponential         ##############################
##############################################################################

# Analytical sol of Activaion Times
def ATexact(inputs):
    if(inputs.shape[1]==1):
        X = inputs[:,0][...,None]
    else:
        raise Error("This case works only in 1D")
    return 1-np.exp(-4.*X)

# Analytical sol of Conduction Velocity
def CVexact(inputs):
    if(inputs.shape[1]==1):
        X = inputs[:,0][...,None]
    else:
        raise Error("This case works only in 1D")
    return (1./4.)*np.exp(4.*X)
