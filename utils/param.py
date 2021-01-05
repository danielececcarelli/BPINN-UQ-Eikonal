#
import json
import os

# local import
from helpers import string_to_bool

"""
Class to handle all the parameters
"""
class param:
    """Initializer"""
    def __init__(self, hp, args):
        self.method = args.method   # method used: SVGD or HMC
        self.architecture = hp["architecture"]  # NN architecture param
        self.experiment = hp["experiment"]  # experiment param
        self.param = hp["param"]    # general param
        self.sigmas = hp["sigmas"]  # sigmas param
        self.utils = hp["utils"]    # utils param

        self.param_method = hp[args.method] # specific param for the selected method

        self._change_string_to_bool()   # convert all the string "True" or "False" to boolean

        # if we have some additional parameters from the command-line
        if(self._length_additional(vars(args)) > 0):
            self._update(vars(args))    # update param overspecified by command-line

        # for these parameters we use the dictionaries specified at the bottom (n_input, pde and dataset_type)
        self.n_input = n_input[self.experiment["dataset"]] # store the dimension input (1D, 2D or 3D)
        self.pde = pde[self.experiment["dataset"]]  # isotropic or anisotropic
        self.dataset_type = dataset_type[self.experiment["dataset"]]    # analytical or dataset

        # dimension of velocity: 1 if isotropic, >1 if anisotropic
        self.n_output_vel = 1
        if(self.experiment["dataset"]=="anisotropic1"):
            self.n_output_vel = 3
        if(self.experiment["dataset"]=="anisotropic2"):
            self.n_output_vel = 2

        # if experiment is analytical, specify the n of domain points
        if(self.dataset_type == "analytical"):
            if self.n_input == 1:
                self.experiment["n_domain"] = 1000
            elif self.n_input == 2:
                self.experiment["n_domain"] = 10000
            else:
                self.experiment["n_domain"] = 1000000

        # check possible errors in parameters
        self._check_parameter()

################################################################################

    """Compute the lenght of additional parameters specified by command-line"""
    def _length_additional(self, args_dict):
        i = 0
        for key in args_dict:
            if(args_dict[key] != None):
                i+=1
        return (i-2)    # the first 2 are mandatory (method and config) and dont count as additional param

    """Change "True" and "False" string to boolean for each bool parameter """
    def _change_string_to_bool(self):
        self.experiment["is_uniform_exact"] = string_to_bool(self.experiment["is_uniform_exact"])
        self.sigmas["data_prior_noise_trainable"] = string_to_bool(self.sigmas["data_prior_noise_trainable"])
        self.sigmas["pde_prior_noise_trainable"] = string_to_bool(self.sigmas["pde_prior_noise_trainable"])
        self.utils["verbose"] = string_to_bool(self.utils["verbose"])
        self.utils["save_flag"] = string_to_bool(self.utils["save_flag"])

    """Update the parameter given by json file using args (overspecification by command-line)"""
    def _update(self, args_dict):
        i = 0
        for key in args_dict:
            if(i > 2):
                if args_dict[key] != None:
                    if key in self.architecture:
                        self.architecture[key] = args_dict[key]
                    elif key in self.experiment:
                        self.experiment[key] = args_dict[key]
                    elif key in self.param:
                        self.param[key] = args_dict[key]
                    elif key in self.sigmas:
                        self.sigmas[key] = args_dict[key]
                    elif key in self.utils:
                        self.utils[key] = args_dict[key]
                    else: #param_method...
                        if key in self.param_method:
                            self.param_method[key] = args_dict[key]
                        else:
                            print("Wrong parameter ", key," for the selected method: ", self.method)
            i+=1

    """Check the parameters"""
    def _check_parameter(self):
        pass

    """Print all the parameters"""
    def print_parameter(self):
        print("Method: ", self.method, " \n ")
        print("architecture: ", self.architecture, " \n ")
        print("experiment: ", self.experiment, " \n ")
        print("param: ", self.param, " \n ")
        print("sigmas: ", self.sigmas, " \n ")
        print("utils: ", self.utils, " \n ")
        print("param_method: ", self.param_method, " \n ")
        print("n_input: ", self.n_input, " \n ")
        print("n_output_vel: ", self.n_output_vel, " \n ")
        print("pde_type: ", self.pde, " \n ")
        print("dataset_type: ", self.dataset_type)

    """Save parameters"""
    def save_parameter(self, path=""):
        with open(os.path.join(path,'param.json'), 'w') as outfile:
            outfile.write("{ \n")

            outfile.write(" \"architecture\": ")
            json.dump(self.architecture, outfile)
            outfile.write(", \n")

            outfile.write(" \"experiment\": ")
            json.dump(self.experiment, outfile)
            outfile.write(", \n")

            outfile.write(" \"param\": ")
            json.dump(self.param, outfile)
            outfile.write(", \n")

            outfile.write(" \"sigmas\": ")
            json.dump(self.sigmas, outfile)
            outfile.write(", \n")

            s = " \""+self.method+"\": "
            outfile.write(s)
            json.dump(self.param_method, outfile)
            outfile.write(", \n")

            outfile.write(" \"utils\": ")
            json.dump(self.utils, outfile)
            outfile.write("\n")

            outfile.write("}")

"""dictionary for input dimension given the dataset used"""
n_input = {
"exponential": 1,
"circle": 2,
"triflag": 2,
"square_with_circle": 2,
"anisotropic1": 2,
"anisotropic2": 2,
"prova3D": 3,
"prolate3D": 3,
"prolate3D_scar_new_version": 3,
"prolate3D_new_version": 3
}


"""dictionary for isotropic or anisotropic pde given the dataset used"""
pde = {
"exponential": "isotropic",
"circle": "isotropic",
"triflag": "isotropic",
"square_with_circle": "isotropic",
"anisotropic1": "anisotropic",
"anisotropic2": "anisotropic",
"prova3D": "isotropic",
"prolate3D": "isotropic",
"prolate3D_scar_new_version": "isotropic",
"prolate3D_new_version": "isotropic"
}


"""dictionary for dataset_type (analytical functions or real dataset from Pykonal) given the dataset used"""
dataset_type = {
"exponential": "analytical",
"circle": "analytical",
"triflag": "dataset",
"square_with_circle": "dataset",
"anisotropic1": "analytical",
"anisotropic2": "analytical",
"prova3D": "dataset",
"prolate3D": "dataset",
"prolate3D_scar_new_version": "dataset",
"prolate3D_new_version": "dataset"
}
