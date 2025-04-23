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
    
from data_collator import DataCollator
from data.vocab import BinVocab


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

def formulate_targets(input_ind, masked_ind):
    formulated_targets = []
        
    device = input_ind.device
    input_shape = input_ind.size()
    masked_shape = masked_ind.size()
        
    input_ind = input_ind.view(-1).tolist()
    input_token = bin_vocab.ind_to_token(input_ind)
    input_chr, input_pos = [token[0] for token in input_token], [token[1] for token in input_token]
        
    input_chr = torch.tensor(input_chr, device=device).view(input_shape)
    input_pos = torch.tensor(input_pos, device=device).view(input_shape)
        
    masked_ind = masked_ind.view(-1).tolist()
    masked_token = bin_vocab.ind_to_token(masked_ind)
    masked_chr, masked_pos = [token[0] for token in masked_token], [token[1] for token in masked_token]
        
    masked_chr = torch.tensor(masked_chr, device=device).view(masked_shape)
    masked_pos = torch.tensor(masked_pos, device=device).view(masked_shape)
        
    for c in range(1, 24):
        bin_num = bin_vocab.bin_num_dict[c]
        target_c = torch.zeros((input_shape[0], bin_num), device=device)
            
        row = torch.where(input_chr==c)[0]
        col = input_pos[torch.where(input_chr==c)] - 1  # -1 because the index starts from 0
        target_c[row, col] = 1
            
        row = torch.where(masked_chr==c)[0]
        col = masked_pos[torch.where(masked_chr==c)] - 1  # -1 because the index starts from 0
        target_c[row, col] = 2
            
        formulated_targets.append(target_c)
                
    return formulated_targets


for i, batch in enumerate(train_loader):
    input_ind = batch["input_ind"]
    masked_ind = batch["masked_ind"]
    break

targets = formulate_targets(input_ind, masked_ind)

print((torch.nonzero(targets[0][0]).view(-1) + 1 == \
    raw_dataset[0]["pos_id"][torch.where(raw_dataset[0]["chr_id"] == 1)[0]]).all())



    