#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

python ../process_all_counts.py \
    --data-source "/lustre/project/Stat/1155223034/atacFormer/data/heart" \
    --pad-value 0 \

