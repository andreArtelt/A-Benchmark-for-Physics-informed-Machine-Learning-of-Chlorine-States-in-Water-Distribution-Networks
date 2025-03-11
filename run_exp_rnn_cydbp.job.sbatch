#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mem=64GB
#SBATCH -e slurm/slurm-%j-%a.err
#SBATCH -o slurm/slurm-%j-%a.out
#SBATCH --ntasks=1
#SBATCH --array=1-286

srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py CY-DBP $SLURM_ARRAY_TASK_ID 0
srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py CY-DBP $SLURM_ARRAY_TASK_ID 1
srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py CY-DBP $SLURM_ARRAY_TASK_ID 2
srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py CY-DBP $SLURM_ARRAY_TASK_ID 3
srun ~/miniconda3/envs/ai4ci/bin/python run_exp_rnn.py CY-DBP $SLURM_ARRAY_TASK_ID 4
