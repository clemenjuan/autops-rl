#!/bin/bash
################################################################################
# SLURM directives##############################################################

# Specify the partition to run the job
#SBATCH -p lrz-hgx-h100-94x4
# #SBATCH -p lrz-dgx-a100-80x8
# #SBATCH -p lrz-dgx-a100-80x4
# #SBATCH -p lrz-v100x2
# #SBATCH -p lrz-hpe-p100x4

# Request GPUs or CPUs
#SBATCH --gres=gpu:1

# Specify the file for outputs
#SBATCH -o test.out

# Specify the file for errors
#SBATCH -e test.err

# Set a time limit for the job (maximum of 2 days in LRZ AI Systems: --time=2-00:00:00)
#SBATCH --time=0-00:10:00

# Send an email when the job ends
# #SBATCH --mail-type=END

# Email address for notifications
# #SBATCH --mail-user=clemente.juan@tum.de

# Container
#SBATCH --container-image='/dss/dsshome1/05/ge26sav2/autops-rl/nvidia+pytorch+25.02-py3.sqsh'
#SBATCH --container-mounts=/dss/dsshome1/05/ge26sav2/autops-rl:/workspace


################################################################################
# Execution commands ###########################################################

nvidia-smi
python3 hello_world.py