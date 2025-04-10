import sys
sys.path.append("../..")

from typing import Optional
import torch
from torch import nn, Tensor
import pandas as pd
from data.vocab import BinVocab
from model.model_sampling import *

# %load_ext autoreload
# %autoreload 2


# load the bin vocab
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_vocab = BinVocab(bin_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
bin_emb_dim = 64
eos_emb_dim = 512
d_model = 64

# bin_embedding = nn.Embedding(len(bin_vocab.vocab), bin_emb_dim)

# decoder = Decoder(vocab=bin_vocab,
#                   bin_embedding=bin_embedding,
#                   eos_emb_dim=eos_emb_dim,
#                   bin_emb_dim=bin_emb_dim,
#                   d_model=d_model,
#                   ).to(device)


# eos_emb = torch.randn(batch_size, 23, 512)
# eos_emb = eos_emb.to(device)

# p = 0.005
# probabilities = [1-p, 0.85 * p, 0.15 * p]  # Probabilities for sampling 0, 1, 2
# formulated_targets = [torch.multinomial(torch.tensor(probabilities, device=device), 
#                                         bin_vocab.bin_num_dict[i] * batch_size, 
#                                         replacement=True).view(batch_size, bin_vocab.bin_num_dict[i])
#                       for i in range(1, 24)]

# predictions = decoder.forward(eos_emb,
#                       use_bin_proj=False,
#                       use_eos_proj=True,
#                       sampling_prop=1,
#                       formulated_targets=formulated_targets)

# load dataset
from data_collator import DataCollator
from datasets import load_dataset

cls_prefix_datatable = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet"
cache_dir = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cache"

raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
            )

raw_dataset = raw_dataset.with_format("torch")

# create the data collator
collator = DataCollator(vocab=bin_vocab,
                        max_input_len=6800,
                        max_masked_len=3000,
                        masked_ratio=0.15)

# train_sampler = RandomSampler(raw_dataset)
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

train_sampler = SequentialSampler(raw_dataset)

train_loader = DataLoader(
    raw_dataset,
    batch_size=8,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)

for i, batch in enumerate(train_loader):
    data_dict = batch
    break

data_dict = {k: v.to(device) for k, v in data_dict.items()}

model = TransformerModel(
    vocab=bin_vocab,
    d_model=512,
    nhead=8,
    d_hid=512,
    nlayers=12,
    use_dna_emb=False,
    use_fast_transformer=True,
    use_class_emb=True,
).to(device)

from torch.cuda.amp import autocast
with autocast(enabled=True):
    output = model(data_dict, decoder_prop=10)

output
