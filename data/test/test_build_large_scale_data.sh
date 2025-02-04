#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

python ../build_large_scale_data.py \
    --input-dir "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart" \
    --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/var_open_cells_23chr.txt"