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

export BASE_MODEL=/users/PAS2836/zheng2545/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a
# export BASE_MODEL=/users/PAS2836/zheng2545/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
# export BASE_MODEL=/users/PAS2836/zheng2545/.cache/huggingface/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a

export DATA_DIR="/users/PAS2836/zheng2545/ih_rl/dataset/qwen3-jailbreak-v1+"
# export DATA_DIR="/users/PAS2836/zheng2545/ih_rl/dataset/phi-4-mini-jailbreak-v1+"

export EXPERIMENT_NAME="Qwen3-4B-GRPO-01R-2048-v1+"
# export EXPERIMENT_NAME="Qwen3-8B-GRPO-01R-2048-v1+"
# export EXPERIMENT_NAME="Phi-4-mini-GRPO-01R-2048-v1+"

export HYDRA_FULL_ERROR=1
export TEMPLATED_INPUT=1

bash ~/ih_rl/scripts/train_grpo.sh