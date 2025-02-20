#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

python ../process_all_counts.py \
    --data-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart" \
    --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"

