#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

# heart, kidney_left, kidney_right, liver, lung_right, pancreas, spleen, small_intestine, large_intestine
python ../build_large_scale_data.py \
    --input-dir "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/kidney_left" \
    --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"