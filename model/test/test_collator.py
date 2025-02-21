import sys
sys.path.append("../")
sys.path.append("../..")

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets

from data.vocab import BinVocab
from data_collator import DataCollator

# load dataset
cls_prefix_datatable = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet"
cache_dir = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cache"

raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
            )

raw_dataset = raw_dataset.with_format("torch")

# load the bin vocab
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_vocab = BinVocab(bin_file)

# create the data collator
collator = DataCollator(vocab=bin_vocab, max_input_len=6800, max_masked_len=3000,
                        masked_ratio=0.15)

# train_sampler = RandomSampler(raw_dataset)
train_sampler = SequentialSampler(raw_dataset)

train_loader = DataLoader(
    raw_dataset,
    batch_size=16,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)

# test the speed of the data collator
for batch in tqdm(train_loader, desc="Loading data"):
    pass

for i, batch in enumerate(train_loader):
    if i == 0:
        input_ind = batch["input_ind"]
        masked_ind = batch["masked_ind"]
        print("input shape:", input_ind.shape)
        if collator.masked_ratio > 0:
            print("masked shape:", masked_ind.shape)
        else:
            print(masked_ind)
        break

# use raw_dataset[0] to check the data
chr_id = raw_dataset[0]["chr_id"]
pos_id = raw_dataset[0]["pos_id"]
token = [(int(c), int(p)) for c, p in zip(chr_id, pos_id)]
ind = torch.tensor(bin_vocab.token_to_ind(token))

