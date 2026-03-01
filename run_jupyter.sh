#!/bin/bash
#
#SBATCH --job-name="jupyter_lab_training"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=jupyter_lab.%j.out
#SBATCH --error=jupyter_lab.%j.err

# Load required modules
module load 2025
module load cuda
source .venv/bin/activate

# Activate your conda/virtual environment if needed
# conda activate nlp
# Or if using venv:
# source nlp_env/bin/activate

# Get the hostname and port
JUPYTER_PORT=$(shuf -i 8888-9999 -n 1)
JUPYTER_HOST=$(hostname)

# Start Jupyter Lab
echo "Starting Jupyter Lab on ${JUPYTER_HOST}:${JUPYTER_PORT}"
echo "To access Jupyter Lab, run this command on your local machine:"
echo "ssh -N -L ${JUPYTER_PORT}:${JUPYTER_HOST}:${JUPYTER_PORT} ${USER}@login.delftblue.tudelft.nl"
echo ""

jupyter lab --no-browser --port=${JUPYTER_PORT} --ip=${JUPYTER_HOST}
