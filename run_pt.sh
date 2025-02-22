#!/bin/bash
cd /users/s1155184322/projects/atacFormer

MAX_INPUT_LEN=6800
MAX_MASKED_LEN=3000
LOG_INTERVAL=100
per_proc_batch_size=28

torchrun --nproc_per_node=1 --master_port=29501 pretrain.py \
    --data-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet" \
    --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt" \
    --save-dir "./save/test" \
    --max-input-len $MAX_INPUT_LEN \
    --max-masked-len $MAX_MASKED_LEN \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 1 \
    --masked-ratio 0.0 \
    --log-interval $LOG_INTERVAL \
    --no-cls \
    --fp16 \
    # --use-dna-emb \
    # --use-dna-encoder \
    # --dna-emb-dim 1280 \
    # --dna-emb-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.npy"