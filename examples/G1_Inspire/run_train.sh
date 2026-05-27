# HF_HUB_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=3 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /mnt/data/share/model/GR00T/GR00T-N1.7-3B \
    --dataset-path /mnt/data/fanzhuoyao/data_library/gr00t_lerobot_dataset_v2.1/grasp_black_bottle_cg_0320_right_arm_joints_rel \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1_Inspire/g1_inspire_right_arm_rel_config.py \
    --vlm-model-path /mnt/data/share/model/Cosmos-Reason2-2B \
    --num-gpus 1 \
    --output-dir ./outputs_cg/right-arm_rel-joints_64-chunk \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 30000 \
    --use-wandb \
    --wandb-project n1d7_sft \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8