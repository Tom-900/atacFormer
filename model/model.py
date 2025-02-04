# modified from https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py 

import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange
import warnings

try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    warnings.warn("flash_attn is not installed")
    flash_attn_available = False
    
try:
    from performer_pytorch import Performer
    performer_available = True
except ImportError:
    warnings.warn("performer_pytorch is not installed")
    performer_available = False
    
from fast_transformers.masking import LengthMask
from .layers import FastTransformerEncoderWrapper, TransformerEncoderLayer, FlashTransformerEncoderLayer
from .utils import chr_pos_to_idx


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int, # embedding dim 
        nhead: int, # number of heads in the multiheadattention models
        d_hid: int, # the dimension of the feedforward network model
        nlayers: int, # the number of layers in the transformer encoder
        dropout: float = 0, # dropout rate
        cell_emb_style: str = "cls", # cell embedding style
        use_fast_transformer: bool = False, # whether to use fast transformer
        fast_transformer_backend: str = "performer", # fast transformer backend
        pre_norm: bool = False, # norm layer before/after the attention layer
        use_dna_encoder: bool = True, # whether to use DNA encoder
        dna_emb_dim: int = 512, # DNA embedding dim
        nlayers_dna_enc: int = 2, # number of layers in the DNA encoder
        bottoleneck_dim: int = 64, # bottleneck dim for bin embedding
        n_cls: Tuple[int, int] = (1, 1), # the number of classes for classification (organ, celltype)
        nlayers_cls: int = 3,   # the number of layers in the classifier
        num_bins_list: list[int] = None, # number of bins for each chromosome
    ):
        
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.cell_emb_style = cell_emb_style
        self.norm_scheme = "pre" if pre_norm else "post"
        self.num_bins_list = num_bins_list

        # cell embedding style
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        
        # fast attention
        if use_fast_transformer:
            if not performer_available:
                warnings.warn(
                    "performer_pytorch is not installed."
                    "Installing performer_pytorch is highly recommended."
                )
            
            elif not flash_attn_available:
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        
        if not use_dna_encoder:
            assert d_model == dna_emb_dim, "d_model should be equal to dna_emb_dim when not using DNA encoder"
        self.use_dna_encoder = use_dna_encoder

        # Extracting DNA sequence encoder
        if self.use_dna_encoder:
            self.dna_encoder = DNAEncoder(dna_emb_dim, d_model, num_conv_layers=nlayers_dna_enc, num_linear_layers=nlayers_dna_enc)
        
        # Bin encoder
        self.bin_encoder = BinEmbedding(num_bins_list=self.num_bins_list, embedding_dim=d_model, bottleneck_dim=bottoleneck_dim)

        # Transformer encoder
        if use_fast_transformer:
            # linear transformer
            if fast_transformer_backend == "linear":
                if fast_transformer_backend == "linear":
                    self.transformer_encoder = FastTransformerEncoderWrapper(
                        d_model, nhead, d_hid, nlayers, dropout
                    )
            # flash transformer
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            # performer
            elif fast_transformer_backend == "performer":
                self.transformer_encoder = Performer(
                    dim=d_model,
                    depth=nlayers,
                    heads=nhead,
                    dim_head=d_model // nhead,
                    causal=False,
                    ff_mult=4,
                    nb_features=d_hid,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    reversible=False,
                    ff_glu=False,
                )
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Model decoders
        self.decoder = Decoder(d_model, num_bins_list=self.num_bins_list,
                               bottleneck_dim=bottoleneck_dim,
                               embedding_dim=d_model,
                               embedding_matrix=self.bin_encoder.embedding_matrix)
            
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

    # basic forward method
    def _encode(
        self,
        seq: Tensor, # DNA sequence, (batch_size, seq_len, dna_emb_dim)
        input: Tensor, # bin index, (batch_size, seq_len)
        src_key_padding_mask: Optional[Tensor] = None, # mask for src, (batch_size, seq_len)
    ) -> Tensor:
        
        if self.use_dna_encoder:
            seq_emb = self.dna_encoder(seq)  # (batch, seq_len, d_model)
        else:
            seq_emb = seq
            
        bin_emb = self.bin_encoder(input)  # (batch, seq_len, d_model)
        total_embs = seq_emb + bin_emb
        
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(total_embs.shape[:2], dtype=torch.bool, device=total_embs.device)
        
        output = self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        
        return output  # (batch, seq_len, d_model)
    
    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
        # get cell embedding from the transformer output
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb
    
    def _get_eos_emb_from_layer(self, 
                                input: Tensor, #(batch_size, seq_len)
                                layer_output: Tensor # (batch, seq_len, d_model)
                                ):
        # get the embedding corresponding to 23 <eos> tokens
        eos_emb = torch.zeros(layer_output.size(0), 23, layer_output.size(2), device=layer_output.device)
        for i in range(layer_output.size(0)):
            # eos start index is sum(num_bins_list) + 1, which is in utils.py
            eos_start = torch.where(input[i] == sum(self.num_bins_list) + 1)[0].item()
            eos_emb[i] = layer_output[i][eos_start: eos_start + 23]
            
        return eos_emb
    
    def _formulate_targets(self,
                          input_chr, # (batch_size, seq_len)
                          target_chr, # (batch_size, seq_len)
                          target_pos,  # (batch_size, seq_len)
                         ):
        formulated_targets = []
        for c in range(23):
            target_c = torch.zeros((input_chr.size(0), self.num_bins_list[c]), device=input_chr.device)
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
        
    def forward(
        self,
        seq: Tensor, # DNA sequence, (batch_size, seq_len, dna_emb_dim)
        data_dict: Dict[str, Any], # data_dict from data_collator
        src_key_padding_mask: Optional[Tensor] = None, # mask for src, (batch_size, seq_len)
        use_cls: bool = True, # whether to use classification for <cls> token
    
    ) -> Mapping[str, Tensor]:

        output = {}
        
        # get input and target
        input_chr = data_dict["masked_chr"] # (batch_size, seq_len)
        input_pos = data_dict["masked_pos"]
        target_chr = data_dict["chr"]
        target_pos = data_dict["pos"]
        
        # input with shape (batch_size, seq_len_), seq_len_ is the length of the input without <mask>
        input = chr_pos_to_idx(input_chr, input_pos, self.num_bins_list, special_to_zero=False) # (batch_size, seq_len_)
        if src_key_padding_mask is None:
            src_key_padding_mask = input.eq(sum(self.num_bins_list) + 24) # sum(self.num_bins_list) + 24 is for <pad> in utils.py
        
        # embedding and transformer
        transformer_output = self._encode(seq, input, src_key_padding_mask=src_key_padding_mask)
        
        eos_emb = self._get_eos_emb_from_layer(input, transformer_output) # shape: (batch, 23, d_model)
        formulated_targets = self._formulate_targets(input_chr, target_chr, target_pos) # list[(batch, num_bins[i]) for i in range(23)]
        
        # decoder output
        predictions = self.decoder(eos_emb) # list[(batch, num_bins[i], 3) for i in range(23)]
        output["predictions"] = predictions  
        output["formulated_targets"] = formulated_targets
        
        # get cell embedding, counts is the weights for weighted pooling only  
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb
        
        # classification output
        if use_cls:
            cls_output = self.cls_decoder(cell_emb)
            output["og_logits"] = cls_output["og_logits"]
            output["ct_logits"] = cls_output["ct_logits"]

        return output


class DNAEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        d_model: int,
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
    ):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1))
            conv_layers.append(nn.LeakyReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
                
        # Fully connected layer
        linear_layers = []
        if num_linear_layers > 2:
            for _ in range(num_conv_layers - 2):
                linear_layers.append(nn.Linear(embedding_dim, embedding_dim))
                linear_layers.append(nn.LeakyReLU())
        if num_linear_layers > 1:
            linear_layers.append(nn.Linear(embedding_dim, embedding_dim // 2))
            linear_layers.append(nn.LeakyReLU())
            linear_layers.append(nn.Linear(embedding_dim // 2, d_model))
            linear_layers.append(nn.LeakyReLU())
        else:
            linear_layers.append(nn.Linear(embedding_dim, d_model))
            linear_layers.append(nn.LeakyReLU())
        self.linear_layers = nn.Sequential(*linear_layers)  

        # Layer normalization
        self.enc_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len) for Conv1d
        # if x.dtype == torch.float16:
        #     self.conv_layers = self.conv_layers.to(dtype=torch.float16)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len) for Conv1d
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, embedding_dim) for FC
        x = self.linear_layers(x)
        x = self.enc_norm(x)
        return x # (batch, seq_len, d_model)


class BinEmbedding(nn.Module):
    def __init__(self, num_bins_list, embedding_dim, bottleneck_dim):
        super(BinEmbedding, self).__init__()
        
        self.num_bins_list = num_bins_list # the number of bins for each chromosome
        self.num_bins = sum(num_bins_list) # the number of bins (V)
        self.embedding_dim = embedding_dim # Final embedding dimension (d)
        self.bottleneck_dim = bottleneck_dim # Bottleneck dimension (k)

        # First matrix: Embedding lookup (V x k)
        # 25 for <cls>, <eos> (23 chr) and <pad>
        self.embedding_matrix = nn.Embedding(self.num_bins + 25, bottleneck_dim) 
        
        # Second matrix: Projection (k x d)
        self.projection_matrix = nn.Linear(bottleneck_dim, embedding_dim, bias=False)

    def forward(self, input, use_proj=True):
        embeddings = self.embedding_matrix(input)  # (batch_size, seq_len, k)
        if use_proj:
            embeddings = self.projection_matrix(embeddings)  # (batch_size, seq_len, d)
        else:
            assert self.bottleneck_dim == self.embedding_dim, "The bottleneck_dim should be equal to embedding_dim"

        return embeddings
        

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
                chunk_size: Optional[int]=1,
                use_bin_proj: bool=False,
                use_eos_proj: bool=True,
                ):  
        
        assert eos_emb.size(2) == self.embedding_dim, "eos_emb should have the same embedding_dim as the model"
        batch_size = eos_emb.size(0)
        
        if isinstance(chunk_size, int):
            if chunk_size > batch_size:
                chunk_size = batch_size
            else:
                assert batch_size % chunk_size == 0, "batch_size should be divisible by chunk_size"
        elif chunk_size is None:
            chunk_size = batch_size
           
        num_bins_list = [0] + self.num_bins_list
        cum_num_bin_tensor = (1 + torch.cumsum(torch.tensor(num_bins_list), dim=0)).long()
        
        # project eos embedding
        if use_eos_proj or self.embedding_dim != self.d_model:
            eos_emb = self.eos_projection(eos_emb)  # (batch, 23, d_model)
        
        predictions = []
        for c in range(23):
            # get bin embedding on chromosome c
            idx_range = torch.arange(cum_num_bin_tensor[c], cum_num_bin_tensor[c+1], device=eos_emb.device)
            bin_emb_c = self.embedding_matrix(idx_range)  # (bin_len_c, bottleneck_dim)
            if use_bin_proj or self.bottleneck_dim != self.d_model:
                bin_emb_c = self.bin_projection(bin_emb_c)  # (bin_len_c, d_model) 
            bin_emb_c = bin_emb_c.unsqueeze(0).repeat(chunk_size, 1, 1)  # (chunk, bin_len_c, d_model)
            
            bin_len_c = bin_emb_c.size(1)
            predictions_c = []
            
            for i in range(0, batch_size, chunk_size):
                # get eos embedding
                eos_emb_c = eos_emb[i:i+chunk_size, c, :] # (chunk, embedding_dim)
                repeated_eos_emb_c = eos_emb_c.unsqueeze(1).repeat(1, bin_len_c, 1)  # (chunk, bin_len_c, d_model)
                
                # concatenate bin and eos embeddings
                total_emb_c = torch.cat((bin_emb_c, repeated_eos_emb_c), dim=2) # (chunk, bin_len_c, 2*d_model)
                prediction_c = self.fc(total_emb_c)  # (chunk, bin_len_c, 3)
                predictions_c.append(prediction_c)
            
            predictions_c = torch.cat(predictions_c, dim=0)  # (batch, bin_len_c, 3)
            predictions.append(predictions_c)
        
        return predictions
        

class ClsDecoder(nn.Module):
    # Decoder for classification task.
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
            
        n_organ, n_celltype = n_cls
        self.og_head = nn.Linear(d_model, n_organ)
        self.ct_head = nn.Linear(d_model, n_celltype)

    def forward(self, x: Tensor) -> Tensor:
        # x: cls embedding, shape [batch_size, embsize]
        for layer in self._decoder:
            x = layer(x)
        return dict(og_logits=self.og_head(x), ct_logits=self.ct_head(x))

