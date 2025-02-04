#!/bin/bash
cd /users/s1155184322/projects/atacFormer

MAX_LENGTH=5000
LOG_INTERVAL=100
per_proc_batch_size=16

python -m torch.distributed.launch --nproc_per_node=2 pretrain.py \
    --data-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet" \
    --dna-emb-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.npy" \
    --atac-bin-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/var_open_cells_23chr.txt" \
    --save-dir ./save/test \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 1 \
    --mask-ratio 0.0 \
    --dna-emb-dim 1280 \
    --use-dna-encoder True \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --fp16