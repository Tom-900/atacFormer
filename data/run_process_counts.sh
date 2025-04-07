#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

# Define an array of organs
# organs=("kidney_left" "kidney_right" "liver" "lung_right" "pancreas" "spleen" "small_intestine" "large_intestine")
organs=("kidney_right" "liver" "lung_right" "pancreas" "spleen" "small_intestine" "large_intestine")

# Loop through each organ and run the build_large_scale_data.py script
for organ in "${organs[@]}"; do
    python process_all_counts.py \
        --data-source "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/$organ" \
        --bin-file "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
done