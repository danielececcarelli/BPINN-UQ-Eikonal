import numpy as np

##############################################################################
########################         anisotropic 2         ##############################
##############################################################################
def sqrt_function(inputs):
    if(inputs.shape[1]==2):
        X0 = 0.5
        Y0 = 0.5
        X = inputs[:,0][...,None]
        Y = inputs[:,1][...,None]
        vec = 4*(X-X0)**2 + (Y-Y0)**2
    else:
        raise Error("This case works only in 2D")
    return np.sqrt(vec)

# Analytical sol of Activaion Times
def ATexact(inputs):
    return 1-np.exp(-sqrt_function(inputs))

# Let's define the conductivity tensor M = [A, 0]
#                                          [0, B]

# Analytical sol of A
def Aexact(inputs):
    return (1/4)*np.reciprocal(np.exp(-2*sqrt_function(inputs)))
# Analytical sol of B
def Bexact(inputs):
    return np.reciprocal(np.exp(-2*sqrt_function(inputs)))

def CVexact(inputs):
    return np.concatenate((Aexact(inputs),Bexact(inputs)), axis=1)
