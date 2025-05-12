#!/bin/bash

#SBATCH --job-name=slm_ensembles                                                            
#SBATCH --output=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.out                
#SBATCH --error=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.err                 
#SBATCH --partition=a40                                                                       
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=16GB                                                  
#SBATCH --time=06:00:00 
 
# Print the name of the node and job ID for debugging
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules if needed on your cluster
# module load cuda/11.8 cudnn/8.6.0

# Activate the virtual environment - uncomment and adjust the path as needed
# source /scratch/ssd004/scratch/klambert/slm_ensembles/env/bin/activate

# Run script
python -u /h/klambert/slm_ensembles/train.py
