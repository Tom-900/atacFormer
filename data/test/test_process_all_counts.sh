#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")"

python ../process_all_counts.py \
    --data-source "/lustre/project/Stat/1155223034/atacFormer/data/try_with_one_file" \
    --bin-file "/lustre/project/Stat/1155223034/atacFormer/data/bins_5k_table_23chr.txt"

