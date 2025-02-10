import sys
sys.path.append("../")
sys.path.append("../../")
import torch 
import numpy as np
from scipy.stats import rankdata
import pandas as pd
from model import *
from utils import chr_pos_to_idx


example1 = {"id": torch.tensor(184117),
            "chr": torch.tensor([0, 1, 6, 21, 23, 3, 2, 15, 12, 15]),
            "pos": torch.tensor([0, 100, 1001, 2000, 400, 10, 244, 13, 134, 234])}
    
example2 = {"id": torch.tensor(184118),
            "chr": torch.tensor([0, 1, 6, 21, 23, 3]),
            "pos": torch.tensor([0, 100, 1001, 2000, 400, 10])}
    
from data_collator import DataCollator
examples = [example1, example1, example2]
data_collator = DataCollator(do_padding=True, max_length=9, keep_first_n_tokens=1, mlm_probability=0.5)
data_dict = data_collator(examples)

input_chr = data_dict["masked_chr"] # (batch_size, seq_len)
input_pos = data_dict["masked_pos"]
target_chr = data_dict["chr"]
target_pos = data_dict["pos"]

bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
num_bins_list = []
for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))

input = chr_pos_to_idx(input_chr, input_pos, num_bins_list, special_to_zero=False)

def formulate_targets(input_chr, # (batch_size, seq_len)
                      target_chr, # (batch_size, seq_len)
                      target_pos,  # (batch_size, seq_len)
                      ):
    formulated_targets = []
    for c in range(23):
        target_c = torch.zeros((input_chr.size(0), num_bins_list[c]), device=input_chr.device)
        for i in range(input_chr.size(0)):
            mask_ic = torch.where(target_chr[i] == c + 1)[0]
            if torch.sum(mask_ic) == 0:
                continue
            else:
                mask_ic_1 = mask_ic[torch.where(input_chr[i][mask_ic] == target_chr[i][mask_ic])[0]]
                mask_ic_2 = mask_ic[torch.where(input_chr[i][mask_ic] != target_chr[i][mask_ic])[0]]
                if torch.sum(mask_ic_1) > 0:
                    target_c[i][target_pos[i][mask_ic_1] - 1] = 1
                if torch.sum(mask_ic_2) > 0:
                    target_c[i][target_pos[i][mask_ic_2] - 1] = 2
        formulated_targets.append(target_c)
            
    return formulated_targets

formulated_targets = formulate_targets(input_chr, target_chr, target_pos)