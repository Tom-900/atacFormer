import argparse
from pathlib import Path
import sys
from datasets import Dataset, load_dataset
import os
sys.path.insert(0, "../")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-source",
    type=str,
    required=True,
    # default=None,
    help='The name of the data source (currently support "scvi" datasets), or the '
    "path to the data file.",
)
parser.add_argument(
    "--cls-value",
    type=int,
    default=0,
    help="The value corresponding to <cls>.",
)
args = parser.parse_args()

def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr": [args.cls_value] + example["chr"],
            "pos": [args.cls_value] + example["pos"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset

data_dir = [str(scb) for scb in Path(args.data_source).glob("*.scb")]
parquet_files = [str(f) for scb in data_dir for f in Path(scb).glob("*.parquet")]
cache_dir = Path(args.data_source) / "cache"

# load or make the dataset w/ <cls> appended at the beginning
cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
if not cls_prefix_datatable.exists():
    print("preparing cls prefix dataset")
    raw_dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=str(cache_dir),)
    raw_dataset = _map_append_cls(raw_dataset)
    raw_dataset.to_parquet(str(cls_prefix_datatable))
    

    
    