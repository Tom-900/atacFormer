# modified from scGPT: https://github.com/bowang-lab/scGPT/blob/integrate-huggingface-model/scgpt/data_collator.py
import sys
sys.path.append("../")
sys.path.append("../..")

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np
from data.vocab import BinVocab


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks atac bin values.

    Args:
        vocab (:obj:`BinVocab`): the vocabulary.
        masked_ratio (:obj:`float`): the probability of masking with MLM.
        max_length (:obj:`int`): the maximum length of the input sequences.
        max_length_ (:obj:`int`): the maximum length of the masked sequences.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from masking. Default to 1.
        keep_last_n_tokens (:obj:`int`): the number of tokens at the end
            of the sequence to keep unchanged from masking. Default to 23.
    """
    vocab: BinVocab
    masked_ratio: float = 0.15
    max_input_len: int = 6800
    max_masked_len: int = 3000
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    keep_last_n_tokens: int = 23

    def __post_init__(self):

        if isinstance(self.masked_ratio, float):
            if self.masked_ratio < 0 or self.masked_ratio >= 1:
                raise ValueError("`masked_ratio` must be between 0 and 1.")
        else:
            raise ValueError("`masked_ratio` must be float.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]


    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
             'chr_id': tensor([0, 1, 3, ..., 0, ..., 0]),
             'pos_id': tensor([0, 1000, 10001, 2, ..., 24])}

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'input_ind': tensor([batch_size, max_input_len]),
                 'masked_ind': tensor([batch_size, max_masked_len])}
        """

        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        device = examples[0]["chr_id"].device

        # get the max input length / masked length
        _max_len = max(len(example["chr_id"]) for example in examples)
        
        _max_input_len = _max_len - int((_max_len - self.keep_first_n_tokens - self.keep_last_n_tokens) * self.masked_ratio)
        
        
        if self.masked_ratio > 0:
            if _max_input_len > self.max_input_len:
                #Restrict the max_input_len to self.max_input_len
                max_input_len = self.max_input_len
                max_masked_len = min(_max_len - self.max_input_len, self.max_masked_len)
            else:
                max_input_len = _max_input_len
                max_masked_len = min(int((_max_len - self.keep_first_n_tokens - self.keep_last_n_tokens) \
                    * self.masked_ratio), self.max_masked_len)
        else:
            max_input_len = self.max_input_len
            max_masked_len = 0

        input = []
        masked = [] if self.masked_ratio > 0 else None
        
        for i in range(len(examples)):
            chr_id = examples[i]["chr_id"]
            pos_id = examples[i]["pos_id"]
            
            token = [(int(c), int(p)) for c, p in zip(chr_id, pos_id)]
            # obtain the cell sentence which is comrpising of open bin indices
            ind = torch.tensor(self.vocab.token_to_ind(token)) 
            
            # mask and pad
            input_ind, masked_ind = self._mask(chr_id, ind)
            
            # input
            input_ind = self._sample(input_ind, max_length=max_input_len, keep=True)
            input_ind = self._pad(input_ind, max_length=max_input_len)
            input.append(input_ind)

            # masked
            if self.masked_ratio > 0:
                masked_ind = self._sample(masked_ind, max_length=max_masked_len)
                masked_ind = self._pad(masked_ind, max_length=max_masked_len)
                masked.append(masked_ind) 

        input = torch.stack(input, dim=0).to(device)
        if self.masked_ratio > 0:
            masked = torch.stack(masked, dim=0).to(device)
        
        data_dict = {
            "input_ind": input,
            "masked_ind": masked,
        }

        # add reserved keys
        device = examples[0]["chr_id"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            data_dict[key] = torch.stack(data_, dim=0).to(device)

        return data_dict
    
    def _sample(
        self,
        ind: torch.LongTensor,
        max_length: int,
        keep: bool = False,
    ):
        
        if len(ind) > max_length:
            if keep:
                #candidates for masking
                ind_kept = ind[self.keep_first_n_tokens:-self.keep_last_n_tokens]
                #randomly keep max_length - keep_first_n_tokens - keep_last_n_tokens tokens
                perm_ind = torch.randperm(len(ind_kept))[:max_length - \
                    self.keep_first_n_tokens - self.keep_last_n_tokens]
                ind_kept = ind_kept[perm_ind]
                #length of the final sentence is max_length
                ind = torch.cat([ind[:self.keep_first_n_tokens], ind_kept, ind[-self.keep_last_n_tokens:]])
                return ind
            else:
                perm_ind = torch.randperm(len(ind))[:max_length]
                return ind[perm_ind]
        else:
            return ind

    def _pad(
        self,
        ind: torch.LongTensor,
        max_length: int,
    ):
        device = ind.device
        dtype = ind.dtype
        pad_value = self.vocab.token_name_to_ind("<pad>")
        
        assert len(ind) <= max_length, (
            f"Input length {len(ind)} is greater than max_length {max_length}."
        )
        
        ind = torch.cat(
            [ind, torch.full((max_length - len(ind),), pad_value, dtype=dtype, device=device)]
        )
        return ind

    
    def _mask(
        self, 
        chr_id: torch.LongTensor,
        ind: torch.LongTensor,
        ) -> torch.Tensor:
        
        """
        Mask the atac ind with MLM.
        """
        if self.masked_ratio > 0:
            #For a sentence, masked_num is the number of candidate tokens to be masked
            masked_num = int((len(ind) - self.keep_first_n_tokens - self.keep_last_n_tokens) * self.masked_ratio)
            #masked_num_ is the number of tokens need to be masked under the restricted input_length(length of sentence after masking)
            masked_num_ = len(ind) - self.max_input_len
            masked_num = max(masked_num, masked_num_)
                
            bin_ind = torch.nonzero(chr_id != 0).squeeze() # The index of bin in the cell sentence(excluding special tokens)
            masked_index = torch.randperm(len(bin_ind))[:masked_num]
            masked_index = bin_ind[masked_index]
                
            input_ind = ind[~torch.isin(torch.arange(len(ind)), masked_index)]
            masked_ind = ind[masked_index]
        else:
            input_ind = ind
            masked_ind = None
        
        return input_ind, masked_ind
    