# modified from scGPT: https://github.com/bowang-lab/scGPT/blob/integrate-huggingface-model/scgpt/data_collator.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks atac bin values.

    Args:
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        mask_value (:obj:`int`): the value to fill at the atac postions that
            are masked.
        eos_value (:obj:`int`): the first value to fill at the end of the sequences.
        eos_length (:obj:`int`): the length of the eos values.
        pad_value (:obj:`int`): the value to use for padding the chr and pos
            to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        mlm_probability (:obj:`float`): the probability of masking with MLM.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
    """
    do_padding: bool = True
    mask_value: int = -1
    eos_value: int = -2
    pad_value: int = -3
    eos_length: int = 23
    do_mlm: bool = True
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1

    def __post_init__(self):
        if self.do_padding:
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")
            
        if self.mask_value >= 0:
            raise ValueError("`mask_value` must be negative.")
        if self.eos_value >= 0:
            raise ValueError("`eos_value` must be negative.")
        if self.pad_value >= 0:
            raise ValueError("`pad_value` must be negative.")

        if isinstance(self.mlm_probability, float):
            if self.mlm_probability < 0 or self.mlm_probability >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        elif isinstance(self.mlm_probability, (list, tuple)):
            if min(self.mlm_probability) < 0 or max(self.mlm_probability) >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        else:
            raise ValueError("`mlm_probability` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
             'chr': tensor([1, 1, ..., 23]),
             'pos': tensor([ 1000,  10001, ..., 20000])}

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'chr': tensor([batch_size, seq_length]),
                'pos': tensor([batch_size, seq_length]),
                'masked_chr': tensor([batch_size, seq_length + 23]),
                'masked_pos': tensor([batch_size, seq_length + 23])}
        """

        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        device = examples[0]["chr"].device

        # get the max length
        max_ori_len = max(len(example["chr"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len
        _max_length = _max_length + self.eos_length

        # pad and truncate
        padded_chr = []
        padded_pos = []
        
        for i in range(len(examples)):
            chr = examples[i]["chr"]
            pos = examples[i]["pos"]
             
            chr, pos = self._sample_or_truncate_plus_pad(chr, pos, _max_length, self.eos_value, self.eos_length)  
            padded_chr.append(chr)
            padded_pos.append(pos)

        padded_chr = torch.stack(padded_chr, dim=0).to(device)
        padded_pos = torch.stack(padded_pos, dim=0).to(device)
        
        data_dict = {
            "chr": padded_chr,
            "pos": padded_pos,
        }

        # mask
        if self.do_mlm:
            masked_chr, masked_pos = self._mask(
                padded_chr, padded_pos, self.keep_first_n_tokens
            )
        else:
            masked_chr = padded_chr
            masked_pos = padded_pos
            
        data_dict["masked_chr"] = masked_chr
        data_dict["masked_pos"] = masked_pos

        # add reserved keys
        device = examples[0]["chr"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            data_dict[key] = torch.stack(data_, dim=0).to(device)

        return data_dict
            
    def _eos_pad(
        self,
        chr: torch.LongTensor,
        pos: torch.LongTensor,
        eos_value: int,
        eos_length: int,
    ):
        device = chr.device
        chr = torch.cat(
            [
                chr,
                torch.full(
                    (eos_length,),
                    eos_value,
                    dtype=chr.dtype,
                    device=device,
                ),
            ]
        )
        pos = torch.cat(
            [
                pos,
                torch.full(
                    (eos_length,),
                    eos_value,
                    dtype=chr.dtype,
                    device=device,
                ),
            ]
        )
        
        return chr, pos
    
    def _sample(
        self,
        chr: torch.LongTensor,
        pos: torch.LongTensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        
        device = chr.device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(chr), device=device)[:max_length]
            return chr[indices], pos[indices]

        # keep the first n tokens unchanged
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(chr) - _n, device=device)[:max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return chr[indices], pos[indices]

    def _pad(
        self,
        chr: torch.LongTensor,
        pos: torch.LongTensor,
        max_length: int,
    ):
        device = chr.device
        chr = torch.cat(
            [
                chr,
                torch.full(
                    (max_length - len(chr),),
                    self.pad_value,
                    dtype=chr.dtype,
                    device=device,
                ),
            ]
        )
        pos = torch.cat(
            [
                pos,
                torch.full(
                    (max_length - len(pos),),
                    self.pad_value,
                    dtype=pos.dtype,
                    device=device,
                ),
            ]
        )
        
        return chr, pos
        
    def _sample_or_truncate_plus_pad(
        self,
        chr: torch.LongTensor,
        pos: torch.LongTensor, 
        max_length: int,
        eos_value: int,
        eos_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        
        assert len(chr) == len(pos)
        if len(chr) + eos_length == max_length:
            chr, pos = self._eos_pad(chr, pos, eos_value, eos_length)
            return chr, pos
        if len(chr) + eos_length > max_length:  # sample or truncate
            if self.sampling:
                chr, pos = self._sample(chr, pos, max_length - eos_length)
                return self._eos_pad(chr, pos, eos_value, eos_length)
            else:
                chr, pos = chr[:max_length - eos_length], pos[:max_length - eos_length]
                return self._eos_pad(chr, pos, eos_value, eos_length)
        else:  # pad
            chr, pos = self._eos_pad(chr, pos, eos_value, eos_length)
            return self._pad(chr, pos, max_length)
    
    def _mask(
        self, 
        chr: torch.Tensor,
        pos: torch.Tensor,
        keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        """
        Mask the atac chr/pos with MLM.
        """
        if keep_first_n_tokens > 0:
            chr_, pos_ = self._mask(
                chr[:, keep_first_n_tokens:],
                pos[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([chr[:, :keep_first_n_tokens], chr_], dim=1), \
                torch.cat([pos[:, :keep_first_n_tokens], pos_], dim=1)

        device = chr.device
        shape = chr.shape

        probability_matrix = torch.full(shape, self.get_mlm_probability())
        # set padded postion and eos positions probability to 0
        probability_matrix[chr.eq(self.pad_value)] = 0
        for i in range(self.eos_value - self.eos_length + 1, self.eos_value + 1):
            probability_matrix[chr.eq(i)] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_chr = chr.masked_fill(mask, self.mask_value)
        masked_pos = pos.masked_fill(mask, self.mask_value)
        return masked_chr, masked_pos
    
    def get_mlm_probability(self) -> float:
        """
        Get the mlm probability for the current step.
        """
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            # random choose a probability
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(
                "mlm_probability must be a float or a list of floats, "
                f"but got {self.mlm_probability}."
            )
        

if __name__ == '__main__':
    example1 = {"id": torch.tensor(184117),
               "chr": torch.tensor([0, 1, 6, 21, 23, 3, 2, 15, 12, 15]),
               "pos": torch.tensor([0, 100, 1001, 2000, 400, 10, 244, 13, 134, 234])}
    
    example2 = {"id": torch.tensor(184118),
               "chr": torch.tensor([0, 1, 6, 21, 23, 3]),
               "pos": torch.tensor([0, 100, 1001, 2000, 400, 10])}
    
    examples = [example1, example1, example2]
    data_collator = DataCollator(do_padding=True, max_length=10, keep_first_n_tokens=1, mlm_probability=0.5)
    data_dict = data_collator(examples)
    
    for key, value in data_dict.items():
        print(key)
        print(value)
        print()
        
    tensor = data_dict['masked_chr']
    
    
    