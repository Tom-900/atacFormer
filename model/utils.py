import torch
import pandas as pd


def chr_pos_to_idx(chr, pos, num_bins_list,
                   mask_value: int=-1,
                   eos_value:int=-2,
                   pad_value:int=-3,
                   special_to_zero=True):
    
    assert mask_value < 0, "mask_value should be negative"
    assert eos_value < 0, "eos_value should be negative"
    assert pad_value < 0, "pad_value should be negative"
    
    device = chr.device
    # Convert num_bins_list to a cumulative sum tensor
    num_bin_tensor = torch.tensor(num_bins_list, dtype=torch.int64)
    num_bin_tensor = num_bin_tensor.to(device)
    cum_num_bin_tensor = torch.cumsum(num_bin_tensor, dim=0)
    
    # Get the lengths for each chrom in the tensor. <cls>, <mask>, <eos> and <pad> are not included
    lengths = torch.where(chr > 1, cum_num_bin_tensor[chr - 2], torch.zeros_like(chr))
    
    # Add the positions to the lengths. <mask>, <eos> and <pad> are unchanged
    index = lengths + pos
    
    # remove <mask> from the sentence
    num_masked = torch.sum(index == mask_value, dim=1)
    seq_len = index.size(1) - num_masked.min().item()
    index_new = torch.zeros(index.size(0), seq_len, dtype=torch.int64, device=device)
    for i in range(index.size(0)):
        seq_len_i = torch.sum(index[i] != mask_value)
        pad_vector = torch.full((seq_len - seq_len_i,), pad_value).to(device)
        index_new[i] = torch.cat((index[i][index[i] != mask_value], pad_vector))
        
    if special_to_zero:
        # transform special token id to 0 (DNA embedding)
        index_new = torch.where(index_new == eos_value, torch.zeros_like(index_new), index_new)
        index_new = torch.where(index_new == pad_value, torch.zeros_like(index_new), index_new)
    else:
        # transform special token id to a non-negative id. 
        # <cls>: 0; <pad>: sum(num_bins_list) + C + 1
        # <eos>: sum(num_bins_list) + 1, ..., sum(num_bins_list) + C
        replacement_sequence = sum(num_bins_list) + torch.arange(1, len(num_bins_list) + 1) 
        replacement_sequence = replacement_sequence.to(device)
        index_new[index_new == eos_value] = replacement_sequence.repeat(index_new.size(0))
        index_new = torch.where(index_new == pad_value, sum(num_bins_list) + len(num_bins_list) + 1, index_new) 
    
    return index_new


if __name__ == '__main__':
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
    
    chr, pos = data_dict["masked_chr"], data_dict["masked_pos"]

    bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
    bin_table = pd.read_table(bin_file, header=None)
    bin_ls = bin_table.iloc[:, 0].tolist()
        
    num_bins_list = []
    for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
        num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))
        
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chr = chr.to(device)
    pos = pos.to(device)
            
    index = chr_pos_to_idx(chr, pos, num_bins_list, special_to_zero=True)
    print(index)



    