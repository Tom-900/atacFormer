import sys
sys.path.append("../")
sys.path.append("../../")
import torch 
import numpy as np
from scipy.stats import rankdata
import pandas as pd


example1 = {"id": torch.tensor(184117),
            "chr_id": torch.tensor([0, 1, 6, 21, 23, 3, 2, 15, 12, 15,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "pos_id": torch.tensor([0, 100, 1001, 2000, 400, 10, 244, 13, 134, 234,
                                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])}
    
example2 = {"id": torch.tensor(184118),
            "chr_id": torch.tensor([0, 1, 6, 21, 23, 3,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "pos_id": torch.tensor([0, 100, 1001, 2000, 400, 10,
                                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])}
    
from data_collator import DataCollator
from data.vocab import BinVocab

bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_vocab = BinVocab(bin_file)

examples = [example1, example1, example2]
data_collator = DataCollator(vocab=bin_vocab, max_input_len=30, max_masked_len=3,
                        masked_ratio=0.3)
data_dict = data_collator(examples)

input_ind = data_dict["input_ind"] # (batch_size, seq_len)
masked_ind = data_dict["masked_ind"]

bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
num_bins_list = []
for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))

# formulate the targets
shape = masked_ind.size()
masked_ind = masked_ind.view(-1).tolist()
masked_token = bin_vocab.ind_to_token(masked_ind)
masked_chr, masked_pos = [token[0] for token in masked_token], [token[1] for token in masked_token]
        
masked_chr = torch.tensor(masked_chr).view(shape)
masked_pos = torch.tensor(masked_pos).view(shape)

c = 3
target_c = torch.zeros(shape[0], num_bins_list[c])
row = torch.where(masked_chr==c)[0]
col = masked_pos[torch.where(masked_chr==c)] - 1
target_c[row, col] = 2

torch.nonzero(torch.where(masked_chr==c, masked_pos, torch.tensor(0)))