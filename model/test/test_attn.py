import torch
import torch.nn as nn
import pandas as pd

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_bins_list
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/var_open_cells_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
num_bins_list = []
for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))

# attention layers
embed_dim = 512
num_heads = 1  
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)

# data
query = torch.randn(23, 1, embed_dim).to(device)
key_len = max(num_bins_list)
key_dim = 512
key = torch.randn(23, key_len, key_dim).to(device)

# mask (N * num_heads, L, S) = (23 * 8, 1, key_len)
attn_mask = torch.zeros(23, num_heads, 1, key_len).to(torch.bool)
mask_ls = [torch.cat((torch.zeros(num_bins_list[i]),
                      torch.ones(key_len - num_bins_list[i])), 
                    ).to(torch.bool)
           for i in range(23)]

for i in range(23):
    attn_mask[i] = mask_ls[i].view(1, 1, key_len).repeat(num_heads, 1, 1)
    
attn_mask = attn_mask.view(23 * num_heads, 1, key_len).to(device)

attn_output, attn_output_weights = multihead_attn(
    query,
    key,
    key,
    attn_mask=attn_mask,
    average_attn_weights=False
)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the attention weights
attn_output_weights = attn_output_weights.squeeze().transpose(0, 1) # (num_heads, 23, key_len)

plt.figure(figsize=(15, 10))
sns.heatmap(attn_output_weights[0].detach().numpy(), cmap='coolwarm')
plt.title('Attention Weights Heatmap')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()

loss = attn_output_weights[0, 0].sum()
loss.backward()


