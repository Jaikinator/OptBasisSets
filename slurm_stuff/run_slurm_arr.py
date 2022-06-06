"""
create a executable file for the slurm array job.
"""

from slurm_stuff.array_job_handler import create_dict,  save_to_file
from optb.data.avdata import elw417 as w417
import os

mol = w417
learning_rate = [2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
basis = ["STO-3G,3-21G", "3-21G,cc-pvtz", "cc-pvdz,cc-pvtz", "aug-cc-pVDZ,aug-pc-2"]
method = ["gd", "adam"]
maxiter = 1000000
miniter = 100000
output = "output"
executable = "/home/ajgxa/.conda/envs/OptBasis/bin/python"

in_dict =  {"molecule": mol,
            "basis": basis,
            "step": learning_rate,
            "maxiter": maxiter,
            "miniter": miniter,
            "method": method,
            "output_path": output,
            "diverge": -1,
            "maxdivattempts": 50,
            "get_misc": True,
            "minimize_kwargs": {"f_rtol": 0}}

in_dict = create_dict(**in_dict)
save_to_file(in_dict, "training_input.json")

with open("test_slurm.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name= OPTB \n")
    f.write("#SBATCH -t 0-00:01 \n")
    f.write("#SBATCH -n 1   \n")
    f.write("#SBATCH --mem= 100MB \n")
    f.write("#SBATCH --mail-type=START,END,FAIL   # notifications for job done & fail \n")
    f.write("#SBATCH --mail-user=jacob.schmieder@student.uni-halle.de \n")
    f.write("#SBATCH -x large033,small177,small108 \n")

    f.write("ml anaconda3/5.2.0 \n")
    f.write("ml gcc/7.2.0 \n")
    f.write("source activate OptBasis \n")

    f.write(f"{executable} create_slurm_folder.py slurm_output_%A \n")

    f.write(f"#SBATCH --output=slurm_output_%A/slurm_output._%a.out \n")

    f.write("\n")

    f.write(f"{executable} main_slurm_arr.py -f training_input.json -i $SLURM_ARRAY_TASK_ID")

os.system(f"sbatch --array 0-{len(in_dict)} run_slurm.sh")

