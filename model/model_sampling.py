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
    
import sys
sys.path.append("../")
sys.path.append("../../")

from .layers import FastTransformerEncoderWrapper, TransformerEncoderLayer, FlashTransformerEncoderLayer
from data.vocab import BinVocab


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab: BinVocab, # bin vocabulary
        d_model: int, # embedding dim 
        nhead: int, # number of heads in the multiheadattention models
        d_hid: int, # the dimension of the feedforward network model
        nlayers: int, # the number of layers in the transformer encoder
        dropout: float = 0, # dropout rate
        cell_emb_style: str = "cls", # cell embedding style
        use_fast_transformer: bool = False, # whether to use fast transformer
        fast_transformer_backend: str = "flash", # fast transformer backend
        pre_norm: bool = False, # norm layer before/after the attention layer
        use_class_emb: bool = False, # whether to use class embedding
        use_dna_emb: bool = True, # whether to use DNA embedding
        use_dna_encoder: bool = True, # whether to use DNA encoder
        dna_emb_dim: int = 512, # DNA embedding dim
        nlayers_dna_enc: int = 2, # number of layers in the DNA encoder
        n_eos: int = 23, # the number of <eos> tokens
        bottoleneck_dim: int = 64, # bottleneck dim for bin embedding
        decoder_dim: int = 64, # decoder dim
        n_cls: Tuple[int, int] = (1, 1), # the number of classes for classification (organ, celltype)
        nlayers_cls: int = 3,   # the number of layers in the classifier
    ):
        
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.cell_emb_style = cell_emb_style
        self.norm_scheme = "pre" if pre_norm else "post"
        self.n_eos = n_eos

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
        
        if use_dna_emb:
            if not use_dna_encoder:
                assert d_model == dna_emb_dim, "d_model should be equal to dna_emb_dim when not using DNA encoder"
            else:
                self.dna_encoder = DNAEncoder(dna_emb_dim,
                                          d_model,
                                          num_conv_layers=nlayers_dna_enc,
                                          num_linear_layers=nlayers_dna_enc,
                                          )
        self.use_dna_emb = use_dna_emb
        self.use_dna_encoder = use_dna_encoder
        
        # Class embedding
        if use_class_emb:
            self.class_embedding = ClassEmbedding(class_dim=d_model)
        else:
            self.class_embedding = None
        self.use_class_emb = use_class_emb
            
        # Bin vocab
        self.bin_vocab = vocab
        
        # Bin encoder
        self.bin_encoder = BinEmbedding(vocab=self.bin_vocab,
                                        embedding_dim=d_model,
                                        bottleneck_dim=bottoleneck_dim,
                                        )
        
        # Transformer encoder
        if use_fast_transformer:
            # linear transformer
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
        self.decoder = Decoder(vocab=self.bin_vocab,
                               bin_embedding=self.bin_encoder.embedding_matrix,
                               eos_emb_dim=d_model,
                               bin_emb_dim=bottoleneck_dim,
                               d_model=decoder_dim,
                               )
            
        self.cls_decoder = ClsDecoder(d_model,
                                      n_cls,
                                      nlayers=nlayers_cls,
                                      )
    
    # get cell embedding from the transformer output
    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
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

    # basic forward method
    def _encode(
        self,
        input_ind: Tensor, # bin index, (batch_size, seq_len)
        dna_emb: Optional[Tensor] = None, # DNA sequence, (batch_size, seq_len, dna_emb_dim)
        input_class: Optional[Tensor] = None, # chr index, (batch_size, seq_len)
        src_key_padding_mask: Optional[Tensor] = None, # mask for src, (batch_size, seq_len)
    ) -> Tensor:
        
        # DNA embedding
        if self.use_dna_emb:
            assert dna_emb is not None, "dna_emb is required when use_dna_emb is True"    
            if self.use_dna_encoder:
                dna_emb = self.dna_encoder(dna_emb)  # (batch, seq_len, d_model)
            else:
                dna_emb = dna_emb
          
        # Bin embedding  
        bin_emb = self.bin_encoder(input_ind)  # (batch, seq_len, d_model)
        
        # Class embedding
        if self.use_class_emb:
            assert input_class is not None, "input_chr is required when use_class_emb is True"
            class_emb = self.class_embedding(input_class)
        
        # total embedding
        total_embs = dna_emb + bin_emb if self.use_dna_emb else bin_emb
        total_embs = total_embs + class_emb if self.use_class_emb else total_embs
        
        # mask for input
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(total_embs.shape[:2], dtype=torch.bool, device=total_embs.device)
        
        # transformer encoder
        output = self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        return output
        
    def forward(
        self,
        data_dict: Dict[str, Any], # data_dict from data_collator
        dna_emb: Optional[Tensor] = None, # DNA sequence, (batch_size, seq_len, dna_emb_dim)
        src_key_padding_mask: Optional[Tensor] = None, # mask for src, (batch_size, seq_len)
        use_cls: bool = True, # whether to use classification for <cls> token
        decoder_prop: float = 1, # sampling proportion for decoder
    ) -> Mapping[str, Tensor]:

        output = {}
        
        # get input and target
        input_ind = data_dict["input_ind"] # (batch_size, seq_len)
        masked_ind = data_dict["masked_ind"]
        
        # get chr and pos
        input_chr, input_pos, masked_chr, masked_pos = self._extract_chr_pos(input_ind, masked_ind)
        input_class = self._get_class_emb_input(input_ind, input_chr) # (batch_size, seq_len)
        
        if src_key_padding_mask is None:
            src_key_padding_mask = input_ind.eq(self.bin_vocab.token_name_to_ind("<pad>")) 
        
        # embedding and transformer
        transformer_output = self._encode(
            input_ind,
            dna_emb,
            input_class=input_class,
            src_key_padding_mask=src_key_padding_mask,
            )
        
        # <eos> embedding
        eos_emb = self._get_eos_emb_from_layer(
            input_ind,
            transformer_output,
            ) # shape: (batch, 23, d_model)
        
        # targets
        formulated_targets = self._formulate_targets(
            input_chr,
            input_pos,
            masked_chr,
            masked_pos,
            ) # list[(batch, num_bins[i]) for i in range(23)]
        
        # decoder output
        predictions, targets = self.decoder(
            eos_emb, # list[(batch, sampled_len, 3) for i in range(23)]
            formulated_targets=formulated_targets,
            sampling_prop=decoder_prop,
            ) 
        
        output["predictions"] = predictions  
        output["formulated_targets"] = targets
        output["sampling_prop"] = decoder_prop
        
        # get cell embedding, counts is the weights for weighted pooling only  
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb
        
        # classification output
        if use_cls:
            cls_output = self.cls_decoder(cell_emb)
            output["og_logits"] = cls_output["og_logits"]
            output["ct_logits"] = cls_output["ct_logits"]

        return output
    
    # get class embedding input
    def _get_class_emb_input(self, input_ind, input_chr: Tensor) -> Tensor:
        # input_chr, input_ind: (batch_size, seq_len)
        input_class = input_chr.clone() # (batch_size, seq_len)
        for i in range(input_ind.size(0)):
            eos_start = torch.where(input_ind[i] == self.bin_vocab.token_name_to_ind("<eos_1>"))[0].item()
            input_class[i][eos_start: eos_start + self.n_eos] = torch.arange(1, self.n_eos + 1)
        return input_class
    
    # get the embedding corresponding to <eos> tokens
    def _get_eos_emb_from_layer(self, 
                                input_ind: Tensor, #(batch, seq_len)
                                output_emb: Tensor # (batch, seq_len, d_model)
                                ):
        eos_emb = torch.zeros(output_emb.size(0), self.n_eos, output_emb.size(2),
                              device=output_emb.device) # (batch, self.n_eos, d_model)
        for i in range(input_ind.size(0)):
            eos_start = torch.where(input_ind[i] == self.bin_vocab.token_name_to_ind("<eos_1>"))[0].item()
            eos_emb[i] = output_emb[i][eos_start: eos_start + self.n_eos]
            
        return eos_emb
    
    def _extract_chr_pos(self, input_ind, masked_ind):
        device = input_ind.device
        input_shape = input_ind.size()
        if masked_ind is not None:
            masked_shape = masked_ind.size()
        
        input_ind = input_ind.view(-1).tolist()
        input_token = self.bin_vocab.ind_to_token(input_ind)
        input_chr, input_pos = [token[0] for token in input_token], [token[1] for token in input_token]
        
        input_chr = torch.tensor(input_chr, device=device).view(input_shape)
        input_pos = torch.tensor(input_pos, device=device).view(input_shape)
        
        if masked_ind is not None:
            masked_ind = masked_ind.view(-1).tolist()
            masked_token = self.bin_vocab.ind_to_token(masked_ind)
            masked_chr, masked_pos = [token[0] for token in masked_token], [token[1] for token in masked_token]
            
            masked_chr = torch.tensor(masked_chr, device=device).view(masked_shape)
            masked_pos = torch.tensor(masked_pos, device=device).view(masked_shape)
        else:
            masked_chr = None
            masked_pos = None
            
        return input_chr, input_pos, masked_chr, masked_pos
        
    
    def _formulate_targets(self, input_chr, input_pos, masked_chr, masked_pos):
        formulated_targets = []
        device = input_chr.device
        input_shape = input_chr.size()
        
        for c in range(1, 1 + self.n_eos):
            bin_num = self.bin_vocab.bin_num_dict[c]
            target_c = torch.zeros((input_shape[0], bin_num), device=device)
            
            row = torch.where(input_chr==c)[0]
            col = input_pos[torch.where(input_chr==c)] - 1  # -1 because the index starts from 0
            target_c[row, col] = 1
            
            if masked_chr is not None:
                row = torch.where(masked_chr==c)[0]
                col = masked_pos[torch.where(masked_chr==c)] - 1  # -1 because the index starts from 0
                target_c[row, col] = 2
            
            formulated_targets.append(target_c)
                
        return formulated_targets
    

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
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len) for Conv1d
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, embedding_dim) for FC
        x = self.linear_layers(x)
        x = self.enc_norm(x)
        return x 


class BinEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim, bottleneck_dim, padding_idx=None):
        super(BinEmbedding, self).__init__()
        self.bin_vocab = vocab
        self.embedding_dim = embedding_dim # Final embedding dimension (d)
        self.bottleneck_dim = bottleneck_dim # Bottleneck dimension (k)

        # Embedding lookup (V x k)
        if padding_idx is None:
            padding_idx = self.bin_vocab.token_name_to_ind("<pad>")
        self.embedding_matrix = nn.Embedding(len(self.bin_vocab.vocab), bottleneck_dim, padding_idx=padding_idx) 
        
        # Projection (k x d)
        self.projection_matrix = nn.Linear(bottleneck_dim, embedding_dim, bias=False)

    def forward(self, input, use_proj=True):
        embeddings = self.embedding_matrix(input)  # (batch_size, seq_len, k)
        if use_proj:
            embeddings = self.projection_matrix(embeddings)  # (batch_size, seq_len, d)
        else:
            assert self.bottleneck_dim == self.embedding_dim, "The bottleneck_dim should be equal to embedding_dim"

        return embeddings
    
    
class ClassEmbedding(nn.Module):
    def __init__(self, class_dim):
        super(ClassEmbedding, self).__init__()
        self.class_dim = class_dim
        self.class_embedding = nn.Embedding(24, class_dim) # special token + 23 chromosomes

    def forward(self, input):
        # input: (batch_size, 23)
        embeddings = self.class_embedding(input) # (batch_size, len, class_dim)
        return embeddings


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
                use_bin_proj: bool=False,
                use_eos_proj: bool=True,
                sampling_prop: float=1,
                formulated_targets: Optional[Tensor]=None, # list[(batch, num_bins[i]) for i in range(23)]
                ):  
        
        assert eos_emb.size(2) == self.eos_emb_dim, \
            "eos_emb.size(2) should equal to self.eos_emb_dim"
        batch_size = eos_emb.size(0)
        device = eos_emb.device

        # project eos embedding
        if use_eos_proj or self.eos_emb_dim != self.d_model:
            eos_emb = self.eos_projection(eos_emb)  # (batch, 23, d_model)
        
        token_ls = self.bin_vocab.vocab["token"].tolist() # token list for ATAC bins

        predictions = []
        targets = []
        for c in range(23):
            # get bin index in the vocab for chr c
            token_c = [token for token in token_ls if token[0] == c+1]
            ind_c = torch.tensor(self.bin_vocab.token_to_ind(token_c), device=device) # (num_bins[c] + 1, )
            
            # For chromosome c, count the mean nonzero counts for each batch, and determine the sampled length
            target_c = formulated_targets[c] # (batch, num_bins[c])
           
            sampled_indices, target = sample_zeros(target_c, sampling_prop)
            batch_len = [len(idx) for idx in sampled_indices]
                
            sampled_indices = torch.cat(sampled_indices).long() # (total_len, )
            target = torch.cat(target).long() # (total_len, )
            bin_emb_c = self.bin_embedding(ind_c[sampled_indices])  # (total_len, bin_emb_dim)
                
            # project bin embedding
            if use_bin_proj or self.bin_emb_dim != self.d_model:
                bin_emb_c = self.bin_projection(bin_emb_c)  # (total_len, d_model)
                    
            # get eos embedding for chromosome c
            eos_emb_c = eos_emb[:, c, :] # (batch, d_model)
            eos_emb_ls = [eos_emb_c[b, :].unsqueeze(0).repeat(batch_len[b], 1) for b in range(batch_size)]
            eos_emb_c = torch.cat(eos_emb_ls, dim=0)  # (total_len, d_model)
                
            # concatenate bin and eos embeddings
            total_emb_c = torch.cat((bin_emb_c, eos_emb_c), dim=1) # (total_len, 2*d_model)
            prediction_c = self.fc(total_emb_c)  # (total_len, 3)
            
            # append to the list
            predictions.append(prediction_c) 
            targets.append(target)
                
        return predictions, targets
    

def sample_zeros(matrix, sampling_prop=1):
    """
    Args:
        matrix: (B, L) tensor with values in {0, 1, 2}
        sampling_prop: sampling proportion
    Returns:
        indices: [(B, l)] list containing column indices
        values: [(B, l)] list containing corresponding values (0, 1, or 2)
    """
    B, L = matrix.shape
    device = matrix.device

    # Get indices and values of non-zero elements
    nonzero_mask = (matrix != 0)
    nonzero_indices = [torch.nonzero(row, as_tuple=True)[0] for row in nonzero_mask]
    nonzero_values = [matrix[i, idx] for i, idx in enumerate(nonzero_indices)]
    
    # Calculate number of zeros to sample per row
    num_zeros_to_sample = sampling_prop * nonzero_mask.sum(dim=1)  # [B,]

    # Sample positions of zeros
    zero_mask = ~nonzero_mask
    sampled_zero_indices = []
    for i in range(B):
        zero_indices = torch.nonzero(zero_mask[i], as_tuple=True)[0]
        num_samples = min(int(num_zeros_to_sample[i].item()), L-len(nonzero_indices[i]))
        if num_samples > 0:
            # Randomly sample zero positions without replacement
            sampled = zero_indices[torch.randperm(len(zero_indices))[:num_samples]]
            sampled_zero_indices.append(sampled)
        else:
            sampled_zero_indices.append(torch.tensor([], device=device))

    # Combine indices (non-zeros first, then sampled zeros)
    indices = [
        torch.cat([nonzero_idx, zero_idx])
        for nonzero_idx, zero_idx in zip(nonzero_indices, sampled_zero_indices)
    ]
    
    # Combine corresponding values (non-zero values first, then zeros)
    values = [
        torch.cat([nonzero_val, torch.zeros(len(zero_idx), device=device)])
        for nonzero_val, zero_idx in zip(nonzero_values, sampled_zero_indices)
    ]

    return indices, values
        

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

