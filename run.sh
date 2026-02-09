#!/usr/bin/bash
set -e

# Qwen3-Omni + LoRA: MIntRec2.0 In-scope Intent Recognition
# --modality: 'text' (텍스트 유니모달) | 'multimodal' (텍스트+비디오)

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python run.py \
    --modality 'text' \
    --text_data_path './text_data' \
    --video_data_path './video_data' \
    --model_name 'Qwen/Qwen3-Omni-30B-A3B-Instruct' \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_epochs 10 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 8 \
    --grad_clip 1.0 \
    --val_start_epoch 2 \
    --wait_patience 3 \
    --eval_monitor 'acc' \
    --top_k_checkpoints 3 \
    --num_video_frames 8 \
    --context_len 1 \
    --max_new_tokens 256 \
    --output_dir 'outputs' \
    --results_file 'results.csv' \
    --save_model \
    --seeds 0 \
    --log_dir 'logs'
