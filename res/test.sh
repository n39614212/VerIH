#!/bin/bash

export N_GPUS=2
export BASE_MODEL=/users/PAS2836/zheng2545/ih_zero/res/checkpoints/IH/countdown-QWen-Instruct-format0.1-question0.3/actor/global_step_200
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME="Test"
export VLLM_ATTENTION_BACKEND=XFORMERS
export TEST_MODE=1

scan_path="/users/PAS2836/zheng2545/ih_zero/"
for dir in "$scan_path"/data*/; do
    if [ -d "$dir" ]; then
        export DATA_DIR="$(realpath "$dir")"
        echo "data=$DATA_DIR"

        bash ~/ih_zero/scripts/train_tiny_zero.sh 2>&1 | grep "val/test_score/countdown"
    fi
done