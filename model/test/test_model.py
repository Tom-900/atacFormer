import torch
from torch import nn, Tensor
import pandas as pd
import sys
sys.path.append("../")

from ..model import *
from ..data_collator import DataCollator

from model import *
from data_collator import DataCollator

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# general setting
batch_size = 16
max_len = 5000
length = torch.randint(100, 4900, (batch_size, ))
masked_ratio = 0.3
d_model = 512

id = length
target_chr = [torch.cat((torch.zeros((1, )),
                         torch.randint(1, 24, (l, )))).long() 
              for l in length]
target_pos = [torch.cat((torch.zeros((1, )), 
                         torch.randint(1, 9000, (l, )))).long() 
              for l in length]
examples = [{"id": id[i], "chr": target_chr[i], "pos": target_pos[i]} for i in range(batch_size)]

data_collator = DataCollator(do_padding=True, max_length=max_len, keep_first_n_tokens=1, mlm_probability=0.5)
data_dict = data_collator(examples)
data_dict = {key: value.to(device) for key, value in data_dict.items()}

# DNA embedding
num_masked = torch.sum(data_dict["masked_chr"] == -1, dim=1)
max_len_ = data_dict["masked_chr"].size(1) - num_masked.min().item()
seq = torch.normal(0, 1, (batch_size, max_len_, 512), dtype=torch.float16)
seq = seq.to(device)

# bin list
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
bin_total_counts = bin_table.iloc[:, 1].tolist()                      
num_bins_list = []
for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))

# model
model = TransformerModel(d_model=d_model,
                        nhead=8,
                        nlayers=12,
                        d_hid=d_model,
                        n_cls=(10, 15),
                        nlayers_dna_enc=1,
                        use_fast_transformer=True,
                        fast_transformer_backend="flash",
                        num_bins_list=num_bins_list, 
                        dna_emb_dim=512,
                        ).to(device)

with torch.cuda.amp.autocast(enabled=True):
    output = model(seq, data_dict)