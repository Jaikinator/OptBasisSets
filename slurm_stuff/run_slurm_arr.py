"""
create a executable file for the slurm array job.
"""

from slurm_stuff.array_job_handler import create_dict,  save_to_file
from optb.data.avdata import elw417 as w417
import os
import sys
from time import localtime

########################################################################################################################
# setup executables
########################################################################################################################
startdate = localtime()
executable = sys.executable
shrunf = "run_slurm.sh"
slurm_outfolder = f"slurm_out/{startdate.tm_year}_{startdate.tm_mon}_{startdate.tm_mday}"
job_name = "OPTB"
mail = "jacob.schmieder@student.uni-halle.de" # mail address for slurm
exclude = "large033,small177,small108"
conda_version = "anaconda3/5.2.0"
conda_env = "OptBasis"
gcc_version = "gcc/7.3.0"

input_file_name = "training_input.json"
execute_file_name = "main_slurm_arr.py"
########################################################################################################################
# setup resources
########################################################################################################################
runtime = "0-00:01"  # days-hours:minutes:seconds
cpu_per_task = 8 # number of cpu per task
mem_per_task = "16G" # memory per task



#create output folder structure if it does not exist
if not os.path.exists(slurm_outfolder):
    os.makedirs(slurm_outfolder)

print(f"folder created {slurm_outfolder}")

mol = w417
learning_rate = [2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
basis = ["STO-3G,3-21G", "3-21G,cc-pvtz", "cc-pvdz,cc-pvtz", "aug-cc-pVDZ,aug-pc-2"]
method = ["gd", "adam"]
maxiter = 1000000
miniter = 100000
output = "output"


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
save_to_file(in_dict, input_file_name)

with open("test_slurm.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --job-name={job_name} \n")
    f.write(f"#SBATCH -t {runtime} \n")
    f.write("#SBATCH -n 1   \n")
    f.write(f"#SBATCH --mem={mem_per_task} \n")
    f.write("#SBATCH --mail-type=START,END,FAIL   # notifications for job done & fail \n")
    f.write(f"#SBATCH --mail-user={mail} \n")
    f.write(f"#SBATCH -x {exclude} \n")

    f.write(f"ml {conda_version} \n")
    f.write(f"ml {gcc_version} \n")
    f.write(f"source activate {conda_env}\n")

    f.write(f"#SBATCH --output={slurm_outfolder}/slurm_output_%A_%a.out \n")

    f.write("\n")

    f.write(f"{executable} {execute_file_name} -f {input_file_name} -i $SLURM_ARRAY_TASK_ID")

os.system(f"sbatch --array 0-{len(in_dict) -1} run_slurm.sh")

