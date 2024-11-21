#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=aartelt@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --mem=32GB
#SBATCH -e slurm/slurm-%j-%a.err
#SBATCH -o slurm/slurm-%j-%a.out
#SBATCH --ntasks=1
#SBATCH --array=1-31

srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py Hanoi $SLURM_ARRAY_TASK_ID