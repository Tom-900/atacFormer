import torch
from torch import nn, Tensor
import pandas as pd


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        embedding_dim: int,
        bottleneck_dim: int,
        num_bins_list: list[int],
        embedding_matrix: nn.Embedding,
        num_states: int = 3, # 0: not present, 1: present, 2: masked

    ):
        super(Decoder, self).__init__()
        # fully connected layer for total embedding
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, num_states),
            nn.ReLU(),
        )
        
        # the same as in BinEmbedding
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        
        self.num_bins_list = num_bins_list 
        self.num_bins = sum(num_bins_list) 
        self.embedding_matrix = embedding_matrix
        self.bin_projection = nn.Linear(bottleneck_dim, d_model, bias=False)
        
        # projection for <eos> embedding
        self.eos_projection = nn.Linear(embedding_dim, d_model)
        
    def forward(self, 
                eos_emb: Tensor, # (batch, 23, embedding_dim)
                chunk_size: int=None,
                use_bin_proj: bool=True,
                use_eos_proj: bool=True,
                ):  
        
        assert eos_emb.size(2) == self.embedding_dim, "eos_emb should have the same embedding_dim as the model"
        batch_size = eos_emb.size(0)
        
        if chunk_size is None:
            chunk_size = batch_size
        elif chunk_size > batch_size:
            chunk_size = batch_size
            
        num_bins_list = [0] + self.num_bins_list
        cum_num_bin_tensor = (1 + torch.cumsum(torch.tensor(num_bins_list), dim=0)).long()
        
        # project eos embedding
        if use_eos_proj or self.embedding_dim != self.d_model:
            eos_emb = self.eos_projection(eos_emb)  # (batch, 23, d_model)
        
        predictions = []
        for c in range(23):
            # get bin embedding
            idx_range = torch.arange(cum_num_bin_tensor[c], cum_num_bin_tensor[c+1], device=eos_emb.device)
            bin_emb_c = self.embedding_matrix(idx_range)  # (bin_len_c, bottleneck_dim)
            if use_bin_proj or self.bottleneck_dim != self.d_model:
                bin_emb_c = self.bin_projection(bin_emb_c)  # (bin_len_c, d_model) 
            bin_emb_c = bin_emb_c.unsqueeze(0).repeat(chunk_size, 1, 1)  # (chunk, bin_len_c, d_model)
            
            bin_len_c = bin_emb_c.size(1)
            predictions_c = []
            
            for i in range(0, batch_size, chunk_size):
                end = min(i + chunk_size, batch_size)
                
                # get eos embedding
                eos_emb_c = eos_emb[i:end, c, :] # (chunk, embedding_dim)
                repeated_eos_emb_c = eos_emb_c.unsqueeze(1).repeat(1, bin_len_c, 1)  # (chunk, bin_len_c, d_model)
                
                # concatenate bin and eos embeddings
                total_emb_c = torch.cat((bin_emb_c, repeated_eos_emb_c), dim=2)
                prediction_c = self.fc(total_emb_c)  # (chunk, bin_len_c, 3)
                predictions_c.append(prediction_c)
            
            predictions_c = torch.cat(predictions_c, dim=0)  # (batch, bin_len_c, 3)
            predictions.append(predictions_c)
        
        return predictions
    

bin_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/var_open_cells_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
num_bins_list = []
for chr_name in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr_name]))
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bottleneck_dim = 64
embedding_matrix = nn.Embedding(sum(num_bins_list) + 26, bottleneck_dim).to(device)

batch = 16
eos_emb = torch.randn(batch, 23, 512)
eos_emb = eos_emb.to(device)

decoder = Decoder(num_bins_list=num_bins_list,
                  d_model=64,
                  embedding_dim=512,
                  bottleneck_dim=bottleneck_dim,
                  embedding_matrix=embedding_matrix,
                  ).to(device)

predictions = decoder(eos_emb,
                      chunk_size=1,
                      use_bin_proj=False,
                      use_eos_proj=True)

