export HF_HUB_OFFLINE=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1

CUDA_VISIBLE_DEVICES=7 python gr00t/experiment/launch_finetune.py \
    --base-model-path /mnt/data/share/model/GR00T/GR00T-N1.7-3B \
    --dataset-path /mnt/data/fanzhuoyao/data_library/gr00t_lerobot_dataset_v2.1_20260520_rel/20260520_grasp_place_cup_merge_v2.1 \
    --embodiment-tag G1_D_ARMS_JOINTS \
    --modality-config-path examples/G1_D/g1_d_rel_joints_config.py \
    --vlm-model-path /mnt/data/share/model/Cosmos-Reason2-2B \
    --num-gpus 1 \
    --output-dir ./outputs_g1d_grasp_place_cup_20260520/grasp-place-cup_rel-joints_64-chunk_64-batch \
    --save-total-limit 15 \
    --save-steps 5000 \
    --max-steps 150000 \
    --save-only-model \
    --use-wandb \
    --wandb-project n1d7_g1d_sft-grasp_place_cup \
    --global-batch-size 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.01 \
    --dataloader-num-workers 10 \
    --shard-size 1024 \
    --episode-sampling-rate 0.1