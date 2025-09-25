#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./%j.out

source ~/.bashrc
conda activate ih_rl

unset ROCR_VIS
unset ROCR_VISIBLE_DEVICES
export N_GPUS=2
export ROLLOUT_TP_SIZE=1

export BASE_MODEL=/xxxx/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a
export DATA_DIR="/users/PAS2836/zheng2545/ih_rl/dataset/qwen3-iheval-v0-think"
export EXPERIMENT_NAME="Qwen3-4B-GRPO-01R-2048"

export HYDRA_FULL_ERROR=1
export TEMPLATED_INPUT=1

bash ~/ih_rl/scripts/train_grpo.sh
