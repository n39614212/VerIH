python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /users/PAS2836/zheng2545/ih_rl/res/checkpoints/IH/IH-Qwen3-GRPO-01Reward/global_step_240/actor \
    --target_dir /users/PAS2836/zheng2545/ih_rl/res/checkpoints/IH/IH-Qwen3-GRPO-01Reward/merged_ckpt_240