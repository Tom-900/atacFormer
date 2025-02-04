# build large-scale data in scBank format from a group of AnnData objects
# modified from scGPT: https://github.com/bowang-lab/scGPT/blob/integrate-huggingface-model/data/cellxgene/build_large_scale_data.py
import gc
import json
from pathlib import Path
import argparse
import shutil
import traceback
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
import os
import scanpy as sc

import sys
sys.path.append('open/data_ar/scbank/databank.py')

from preprocess import preprocessor
from databank import DataBank


parser = argparse.ArgumentParser(
    description="Build large-scale data in scBank format from a group of AnnData objects"
)
parser.add_argument(
    "--input-dir",
    type=str,
    required=True,
    help="Directory containing AnnData objects",
)

parser.add_argument(
    "--bin-file",
    type=str,
    required=True,
    help="Path to the bin file",
)

parser.add_argument(
    "--include-files",
    type=str,
    nargs="*",
    help="Space separated file names to include, default to all files in input_dir",
)

parser.add_argument(
    "--filter-bin",
    type=bool,
    default=True,
    help="Whether to filter bins, if True, filter bins that in the bin_file",
)

parser.add_argument(
    "--intersect",
    type=Optional[bool],
    default=True,
    help="Whether to intersect the bins in the bin file with the bins in the adata object",
)

parser.add_argument(
    "--filter-cell-by-bins",
    type=Union[int, bool],
    default=100,
    help="Whether to filter cells by No. of open bins, if :class:`int`, filter cells with counts",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=10000,
    help="The batch size of bins to process the data",
)

args = parser.parse_args()

# extract all the .h5ad files in the input directory
input_dir = Path(args.input_dir)
files = [f for f in input_dir.glob("*.h5ad")]
print(f"Found {len(files)} files in {input_dir}")

# filter files if include_files is provided
if args.include_files is not None:
    files = [f for f in files if f.name in args.include_files]

# preprocessing and transforming into databank object
token_col = "feature_name"
for f in files:
    print(f"\nProcessing {f.name}")
    try:
        adata = sc.read(f, cache=True)
        print(f"originally read {adata.shape} data from {f.name}")
        
        adata = preprocessor(
            adata, 
            filter_bin=args.filter_bin,
            intersect=args.intersect,
            bin_file=args.bin_file,
            filter_cell_by_bins=args.filter_cell_by_bins,
            batch_size=args.batch_size,
        )
        print(f"read {adata.shape} valid data from {f.name}")

        # build databank object
        main_table_key = "X"
        db = DataBank.from_anndata(
            adata,
            to=input_dir / f"{f.stem.split('_')[0]}.scb",
            main_table_key=main_table_key,
            immediate_save=False,
        )
        
        # sync all to disk
        db.meta_info.on_disk_format = "parquet"
        db.sync()

        # clean up and release memory
        del adata
        del db
        gc.collect()
        
    except Exception as e:
        traceback.print_exc()
        warnings.warn(f"failed to process {f.name}: {e}")
        shutil.rmtree(input_dir / f"{f.stem.split('_')[0]}.scb", ignore_errors=True)


        