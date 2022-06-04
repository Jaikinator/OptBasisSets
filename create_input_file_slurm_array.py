import os
import sys
from numpy import savetxt , genfromtxt , save , load ,loadtxt , fromstring
import json
from optb.data.preselected_avdata import elw417 as w417 , elg2 as g2

learning_rate = [2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
basis = ["STO-3G,3-21G", "3-21G,cc-pvtz", "cc-pvdz,cc-pvtz", "aug-cc-pVDZ,aug-pc-2"]
method = ["gd", "adam"]
maxiter = 1000000
miniter = 100000
output = "output"



def create_training_input_dict(basis, molecule, learning_rate , maxiter ,miniter, method, outputpath, diverge = -1.0,
                               maxdivattempts = 50, get_misc = True, minimize_kwargs = {"f_rtol": 0}, save_to_file = True):
    training_input_dict = {}
    i = 0
    for mol in molecule:
        for lr in learning_rate:
            for b in basis:
                for m in method:
                    training_input_dict[i] = {"basis": b.split(",")[0],
                                              "basis_ref": b.split(",")[1],
                                              "molecule": mol,
                                              "step": lr,
                                              "maxiter": maxiter,
                                              "miniter": miniter,
                                              "method": m,
                                              "output_path": outputpath,
                                              "diverge": diverge,
                                              "maxdivattempts": maxdivattempts,
                                              "get_misc": get_misc,
                                              "minimize_kwargs": minimize_kwargs}
                    i += 1
    print(f"{i} jobs created")

    if save_to_file:
        with open("training_input.json", "w") as f:
            json.dump(training_input_dict, f)

    return training_input_dict



def create_training_input_txt(basis, method, learning_rate):
    """
    Creates a txt file with the training input data.
    to load the data use:
    foile = loadtxt("training_input.txt", delimiter=",", dtype = {"names":["mol" , "lr", "basis1", "basis2" , "method" ],
                  "formats":["U50", "f8", "U50", "U50", "U50" ]}).tolist()
    for i in range(len(file)):
        file[i] = list(file[i])
    """
    training_input = []
    for mol in w417:
        for lr in learning_rate:
            for b in basis:
                for m in method:
                    training_input.append([mol, lr, b, m])
    needed_jobs = len(training_input)
    savetxt("training_input.txt", training_input, fmt="%s" , delimiter=",")


create_training_input_dict(basis, w417, learning_rate, maxiter, miniter, method, output, save_to_file = True)


# with open("create_arr_slurm.sh", "w") as f:
#     f.write("#!/bin/bash\n")
#     f.write("#SBATCH --job-name= OPTB \n")
#     f.write("SBATCH -t 3-00:00  \n")
#     f.write("#SBATCH -n 12   \n")
#     f.write("# SBATCH --mail-user=jacob.schmieder@student.uni-halle.de \n")
#     f.write("# SBATCH --mem=24G \n")
#     f.write("# SBATCH --output=slurm_output/OptB_Array_cc-pvdz_cc-pvtz_gd.%A_%a.out \n")
#     f.write("ml anaconda3/5.2.0 \n")
#     f.write("ml gcc/7.2.0 \n")
#     f.write("source activate OptBasis \n")
#     f.write("\n")
#     f.write("/home/ajgxa/.conda/envs/OptBasis/bin/python"
#             " main.py --cres --mol ${molec[$SLURM_ARRAY_TASK_ID]} --basis cc-pvdz cc-pvtz --steps 2e-3 2e-4 2e-5 2e-6 2e-7 2e-8 2e-9 2e-10 2e-11 2e-12 --method gd ")
#
