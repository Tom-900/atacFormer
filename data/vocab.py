import pandas as pd
import json
from typing import Mapping


class BinVocab:
    def __init__(self, bin_file):
        self.bin_file = bin_file
        self.vocab, self.token_to_ind_map, self.token_name_to_ind_map, \
            self.ind_to_token_map, self.token_name_to_token_map = self._create_vocab()
            
        bin_num_ls = [len([i for i in bin_vocab.vocab["token"].tolist() if i[0] == c])\
            for c in range(1, 24)]
        self.bin_num_dict = {i: bin_num_ls[i-1] for i in range(1, 24)}

    def _create_vocab(self):
        bin_table = pd.read_table(self.bin_file, header=None)
        bin_names = bin_table.iloc[:, 0].tolist()
        
        # token names
        chr_id = [23 if i.split(":")[0]=="X" else int(i.split(":")[0])  for i in bin_names]
        pos_id = [int((int(i.split(":")[1].split("-")[0]) - 1) / 5000 + 1) for i in bin_names]
        bin_tokens = [(chr_id[i], pos_id[i]) for i in range(len(chr_id))]

        # special tokens
        special_names = ["<cls>", "<pad>"] + [f"<eos_{i}>" for i in range(1, 24)]
        special_tokens = [(0, i) for i in range(len(special_names))]

        token_names = special_names + bin_names
        tokens = special_tokens + bin_tokens

        ind = [i for i in range(len(token_names))]

        vocab = pd.DataFrame({"token name": token_names, 
                              "token": tokens,
                              "ind": ind})
        
        token_to_ind_map: Mapping[str, int] = {token: i for i, token in enumerate(tokens)}
        token_name_to_ind_map: Mapping[str, int] = {token_name: i for i, token_name in enumerate(token_names)}
        ind_to_token_map: Mapping[int, str] = {i: token for i, token in enumerate(tokens)}
        token_name_to_token_map: Mapping[str, str] = {token_name: token for token_name, token in zip(token_names, tokens)}
        
        return vocab, token_to_ind_map, token_name_to_ind_map, ind_to_token_map, token_name_to_token_map

    def token_to_ind(self, tokens):
        if isinstance(tokens, list):
            indices = []
            for token in tokens:
                if token in self.token_to_ind_map:
                    indices.append(self.token_to_ind_map[token])
                else:
                    raise ValueError(f"Token '{token}' not found in vocabulary.")
            return indices
        else:
            if tokens in self.token_to_ind_map:
                return self.token_to_ind_map[tokens]
            else:
                raise ValueError(f"Token '{tokens}' not found in vocabulary.")
            
    def token_name_to_ind(self, token_names):
        if isinstance(token_names, list):
            indices = []
            for token_name in token_names:
                if token_name in self.token_name_to_ind_map:
                    indices.append(self.token_name_to_ind_map[token_name])
                else:
                    raise ValueError(f"Token name '{token_name}' not found in vocabulary.")
            return indices
        else:
            if token_names in self.token_name_to_ind_map:
                return self.token_name_to_ind_map[token_names]
            else:
                raise ValueError(f"Token name '{token_names}' not found in vocabulary.")

    def ind_to_token(self, indices):
        if isinstance(indices, list):
            tokens = []
            for ind in indices:
                if ind in self.ind_to_token_map:
                    tokens.append(self.ind_to_token_map[ind])
                else:
                    raise ValueError(f"Index '{ind}' not found in vocabulary.")
            return tokens
        else:
            if indices in self.ind_to_token_map:
                return self.ind_to_token_map[indices]
            else:
                raise ValueError(f"Index '{indices}' not found in vocabulary.")
            
    def token_name_to_token(self, token_names):
        if isinstance(token_names, list):
            tokens = []
            for token_name in token_names:
                if token_name in self.token_name_to_token_map:
                    tokens.append(self.token_name_to_token_map[token_name])
                else:
                    raise ValueError(f"Token name '{token_name}' not found in vocabulary.")
            return tokens
        else:
            if token_names in self.token_name_to_token_map:
                return self.token_name_to_token_map[token_names]
            else:
                raise ValueError(f"Token name '{token_names}' not found in vocabulary.")

    def save_vocab(self, file_path):
        vocab_dict = {
            "vocab": self.vocab.to_dict(orient="list"),
            "token_to_ind_map": dict(self.token_to_ind_map),
            "token_name_to_ind_map": dict(self.token_name_to_ind_map),
            "ind_to_token_map": dict(self.ind_to_token_map),
            "token_name_to_token_map": dict(self.token_name_to_token_map),
            "bin_num_dict": dict(self.bin_num_dict)
        }
        with open(file_path, 'w') as f:
            json.dump(vocab_dict, f, indent=4)
            
    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            vocab_dict = json.load(f)
        
        bin_vocab = cls.__new__(cls)
        bin_vocab.vocab = pd.DataFrame(vocab_dict["vocab"])
        bin_vocab.token_to_ind_map = vocab_dict["token_to_ind_map"]
        bin_vocab.token_name_to_ind_map = vocab_dict["token_name_to_ind_map"]
        bin_vocab.ind_to_token_map = vocab_dict["ind_to_token_map"]
        bin_vocab.token_name_to_token_map = vocab_dict["token_name_to_token_map"]
        bin_vocab.bin_num_dict = vocab_dict["bin_num_dict"]
        return bin_vocab
            
    
# example usage     
if __name__ == '__main__':   
    bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
    bin_vocab = BinVocab(bin_file)
    
    tokens = [(0,1), (1,2), (0,3)] * 5000
    bin_vocab.token_to_ind(tokens) 
    
    indices = [0, 1, 2] * 5000
    bin_vocab.ind_to_token(indices)
    
    token_names = ["<cls>", "<pad>", "<eos_1>", "<eos_2>", "<eos_3>"]
    bin_vocab.token_name_to_ind(token_names)
    bin_vocab.token_name_to_token(token_names)

    vocab_path = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bin_vocab.json"
    bin_vocab.save_vocab(vocab_path)

    # reload vocab
    bin_vocab2 = BinVocab.from_json(vocab_path)