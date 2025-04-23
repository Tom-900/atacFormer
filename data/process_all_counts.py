import os
import sys
sys.path.append("../")
import argparse
from pathlib import Path
from datasets import Dataset, load_dataset
from vocab import BinVocab

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
    "--bin-file",
    type=str,
    required=True,
    help="The file containing the ATAC bins.",
)
args = parser.parse_args()

# create bin vocab
bin_vocab = BinVocab(args.bin_file)

def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr_id": [bin_vocab.token_name_to_token("<cls>")[0]] + example["chr_id"],
            "pos_id": [bin_vocab.token_name_to_token("<cls>")[1]] + example["pos_id"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=8,
    )
    return dataset
cpu_count = len(os.sched_getaffinity(0))  # 获取当前机器的 CPU 核心数

# 设置一个合适的默认值（可以修改成合适的数值）
num_proc = min(16, cpu_count)  # 限制最多使用 16 个进程

def _map_append_eos(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr_id": example["chr_id"] + [bin_vocab.token_name_to_token(f"<eos_{i}>")[0] for i in range(1, 24)],
            "pos_id": example["pos_id"] + [bin_vocab.token_name_to_token(f"<eos_{i}>")[1] for i in range(1, 24)],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=8,
        )
    return dataset

data_dir = [str(scb) for scb in Path(args.data_source).glob("*.scb")]
parquet_files = [str(f) for scb in data_dir for f in Path(scb).glob("*.parquet")]
cache_dir = Path(args.data_source) / "cache"

# load or make the dataset w/ <cls> appended at the beginning
cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
if not cls_prefix_datatable.exists():
    print("preparing <cls> prefix and <eos> suffix dataset")
    raw_dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=str(cache_dir),)
    raw_dataset = _map_append_cls(raw_dataset)
    raw_dataset = _map_append_eos(raw_dataset)
    raw_dataset.to_parquet(str(cls_prefix_datatable))
    

    
    