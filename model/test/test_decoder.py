import sys
sys.path.append("../..")

from typing import Optional
import torch
from torch import nn, Tensor
import pandas as pd
from data.vocab import BinVocab


class Decoder(nn.Module):
    def __init__(
        self,
        vocab: BinVocab,
        bin_embedding: nn.Embedding,
        d_model: int,
        eos_emb_dim: int,
        bin_emb_dim: int,
        num_states: int = 3, # 0: not present, 1: present, 2: masked

    ):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.bin_emb_dim = bin_emb_dim
        self.eos_emb_dim = eos_emb_dim
        
        # bin embedding and vocab
        self.bin_embedding = bin_embedding
        self.bin_vocab = vocab
        
        # projection for bin embedding and <eos> embedding
        self.bin_projection = nn.Linear(bin_emb_dim, d_model, bias=False)
        self.eos_projection = nn.Linear(eos_emb_dim, d_model)
        
        # fully connected layer for total embedding
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, num_states),
            nn.ReLU(),
        )
        
    def forward(self, 
                eos_emb: Tensor, # (batch, 23, embedding_dim)
                chunk_size: Optional[int]=1,
                use_bin_proj: bool=False,
                use_eos_proj: bool=True,
                ):  
        
        assert eos_emb.size(2) == self.eos_emb_dim, \
            "eos_emb.size(2) should equal to self.eos_emb_dim"
        batch_size = eos_emb.size(0)
        device = eos_emb.device
        
        # use chunk to save memory
        if isinstance(chunk_size, int):
            if chunk_size > batch_size:
                chunk_size = batch_size
            else:
                assert batch_size % chunk_size == 0, \
                    "batch_size should be divisible by chunk_size"
        elif chunk_size is None:
            chunk_size = batch_size

        # project eos embedding
        if use_eos_proj or self.eos_emb_dim != self.d_model:
            eos_emb = self.eos_projection(eos_emb)  # (batch, 23, d_model)
        
        predictions = []
        token_ls = self.bin_vocab.vocab["token"].tolist()
        for c in range(23):
            # get bin index for chr c
            token_c = [token for token in token_ls if token[0] == c+1]
            ind_c = torch.tensor(self.bin_vocab.token_to_ind(token_c), device=device) # (len_c, )
            if c == 0:
                print(ind_c)
            
            # load bin embedding for chr c
            bin_emb_c = self.bin_embedding(ind_c)  # (len_c, bin_emb_dim)
            print(bin_emb_c.size())
            
            # project bin embedding
            if use_bin_proj or self.bin_emb_dim != self.d_model:
                bin_emb_c = self.bin_projection(bin_emb_c)  # (len_c, d_model) 
            
            # repeat bin embedding
            repeated_bin_emb_c = bin_emb_c.unsqueeze(0).repeat(chunk_size, 1, 1)  # (chunk, len_c, d_model)
            
            len_c = repeated_bin_emb_c.size(1)
            predictions_c = []
            
            for i in range(0, batch_size, chunk_size):
                # get eos embedding for chunk i, chr c
                eos_emb_c = eos_emb[i:i+chunk_size, c, :] # (chunk, embedding_dim)
                repeated_eos_emb_c = eos_emb_c.unsqueeze(1).repeat(1, len_c, 1)  # (chunk, len_c, d_model)

                # concatenate bin and eos embeddings
                total_emb_c = torch.cat((repeated_bin_emb_c, repeated_eos_emb_c), dim=2) # (chunk, len_c, 2*d_model)
                prediction_c = self.fc(total_emb_c)  # (chunk, len_c, 3)
                predictions_c.append(prediction_c)
            
            predictions_c = torch.cat(predictions_c, dim=0)  # (batch, len_c, 3)
            predictions.append(predictions_c)
        
        return predictions
    

# load the bin vocab
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
bin_vocab = BinVocab(bin_file)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
bin_emb_dim = 64
eos_emb_dim = 512
d_model = 64

eos_emb = torch.randn(batch_size, 23, 512)
eos_emb = eos_emb.to(device)

bin_embedding = nn.Embedding(len(bin_vocab.vocab), bin_emb_dim)

decoder = Decoder(vocab=bin_vocab,
                  bin_embedding=bin_embedding,
                  d_model=d_model,
                  bin_emb_dim=bin_emb_dim,
                  eos_emb_dim=eos_emb_dim,
                  ).to(device)

predictions = decoder(eos_emb,
                      chunk_size=1,
                      use_bin_proj=False,
                      use_eos_proj=True)

