"""
This Projekt is based on the dqc package of Kasim, M. F.
https://github.com/diffqc/dqc.git

the optimizer is forked by
https://github.com/xitorch/xitorch.git
"""

from optb.optimize_basis import *
from json import load
import os
import sys
import argparse


########################################################################################################################
# configure torch tensor
########################################################################################################################

torch.set_printoptions(linewidth=200, precision=5)

if __name__ == "__main__":

    ####################################################################################################################
    # setup arg parser throw terminal inputs
    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Optimize Basis')

    ####################################################################################################################
    # configure atomic optb:
    ####################################################################################################################

    parser.add_argument("-f" ,"--file",dest="infile", type = str, metavar="",default = "training_input.json",
                        help='name of the input file.')

    parser.add_argument("-i" ,"--inputID",dest="id", type = str, metavar="",default= "0",
                        help='slurm ID to select index of the input file.')

    ####################################################################################################################
    # parse arguments
    ####################################################################################################################

    args = parser.parse_args()

    ####################################################################################################################
    # load input file
    ####################################################################################################################

    trainingset = load(open(args.infile))
    inputdict = trainingset[args.id]
    Opt = OPTBASIS(**inputdict)
    print(Opt)
    Opt.optimize_basis()
