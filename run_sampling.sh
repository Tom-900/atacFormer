#!/bin/bash
cd /users/s1155184322/projects/atacFormer

MAX_INPUT_LEN=6800
MAX_MASKED_LEN=3000
LOG_INTERVAL=100
per_proc_batch_size=50
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=2 /users/s1155184322/projects/atacFormer/pretrain_sampling.py \
    --data-source "/lustre/project/Stat/s1155184322/datasets/atacFormer" \
    --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt" \
    --save-dir "save/sampling_full_size" \
    --max-input-len $MAX_INPUT_LEN \
    --max-masked-len $MAX_MASKED_LEN \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 2 \
    --warmup-ratio-or-step 0.1 \
    --scheduler-interval 1000 \
    --masked-ratio 0.0 \
    --log-interval $LOG_INTERVAL \
    --no-cls \
    --fp16 \
    --use-class-emb \
    --bin-emb-dim 512 \
    --decoder-dim 512 \