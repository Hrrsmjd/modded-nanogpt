# torchrun --standalone --nproc_per_node=8 train_gpt2.py

#!/bin/bash

# Experiment 1: mask_1=1.0, mask_2=1.0
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M_exp1 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048 \
    --mask_1 0.1 \
    --mask_2 1.0 &

# Experiment 2: mask_1=0.8, mask_2=1.0
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M_exp2 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048 \
    --mask_1 1.0 \
    --mask_2 0.1 &

# Experiment 3: mask_1=1.0, mask_2=0.8
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M_exp3 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048 \
    --mask_1 0.2 \
    --mask_2 1.0 &

# Experiment 4: mask_1=0.8, mask_2=0.8
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M_exp4 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048 \
    --mask_1 1.0 \
    --mask_2 0.2 &

# Wait for all background processes to finish
wait

# CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
#     --input_bin "data/fineweb10B/fineweb_train_*.bin" \
#     --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
#     --output_dir pylog124M_dropout \
#     --model d12 \
#     --batch_size 32 \
#     --sequence_length 1024 \
#     --val_loss_every 128 \
#     --num_iterations 9536 \
#     --weight_decay 0.1 \
#     --learning_rate 0.0018 \
#     --warmup_iters 256 \
#     --warmdown_iters 2048 \
#     --mask_1 1.0 \
#     --mask_2 1.0 &

# wait
