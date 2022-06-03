import os
import sys
from optb.data.preselected_avdata import elw417 as w417 , elg2 as g2

learning_rate = [2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
basis = ["STO-3G 3-21G", "3-21G cc-pvtz", "cc-pvdz cc-pvtz", "aug-cc-pVDZ aug-pc-2"]
method = ["gd", "adam"]

training_input = []

for mol in w417:
    for lr in learning_rate:
        for basis in basis:
            for method in method:
                training_input.append([mol, lr, basis, method])

needed_jobs = len(training_input)

print(sys.argv[0], ":", needed_jobs)
#
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
