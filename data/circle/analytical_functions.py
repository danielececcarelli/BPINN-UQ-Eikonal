import numpy as np

##############################################################################
########################         circle         ##############################
##############################################################################

### build the example case of paper Physics-Informed Neural Networks for Cardiac Activation Mapping
# Analytical sol of Activaion Times
def ATexact(inputs):
    if(inputs.shape[1]==2):
        X = inputs[:,0][...,None]
        Y = inputs[:,1][...,None]
    else:
        raise Exception("This case works only in 2D")
    return np.minimum(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))
# Analytical sol of Conduction Velocity
def CVexact(inputs):
    if(inputs.shape[1]==2):
        X = inputs[:,0][...,None]
        Y = inputs[:,1][...,None]
    else:
        raise Exception("This case works only in 2D")
    mask = np.less_equal(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))
    return mask*1.0 + ~mask*1.0/0.7
