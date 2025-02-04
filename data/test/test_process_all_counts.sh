#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

python ../process_all_counts.py \
    --data-source "/lustre/project/Stat/s1155184322/datasets/atacGPT/HuBMAP/heart" \
    --pad-value 0 \

