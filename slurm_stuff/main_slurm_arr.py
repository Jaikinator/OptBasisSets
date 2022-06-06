"""
This Projekt is based on the dqc package of Kasim, M. F.
https://github.com/diffqc/dqc.git

the optimizer is forked by
https://github.com/xitorch/xitorch.git
"""

from optb.optimize_basis import *
from slurm_stuff.array_job_handler import load_from_file as load
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

    parser.add_argument("-f",dest="file", type = str, metavar="",
                        help='Name of the training input file')

    parser.add_argument("-i", "--id", dest="id", type= str, metavar="",
                        help='slurm ID of the job')

    ####################################################################################################################
    # parse arguments
    ####################################################################################################################

    args = parser.parse_args()
    ###################################################################################################################
    # load input file
    ###################################################################################################################

    input_file = load(args.file)[args.id]

    # split up basis
    input_file["basis"] , input_file["basis_ref"] = input_file["basis"].split(",")



    ####################################################################################################################
    # create output folder to current path
    ####################################################################################################################
    savepath = input_file["output_path"]
    _outf = os.path.join(savepath, "../output")
    if not os.path.exists(_outf):
        os.mkdir(_outf)
    savepath = _outf

    ####################################################################################################################
    # run actual optimization
    ####################################################################################################################
    OPTB = OPTBASIS(**input_file)
    OPTB.optimize_basis()

