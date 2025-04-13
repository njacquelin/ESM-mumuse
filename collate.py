import pandas as pd
import torch

from time import time


class Collate():
    def __init__(self, alphabet):
        self.token_padding = alphabet.tok_to_idx["<pad>"]

    def collate_fn(self, data_dict):
        data = pd.DataFrame(data_dict)
        batch_size = len(data_dict)
        max_len = max(data['seq_length'])

        out = torch.empty((batch_size, max_len, max_len), dtype=torch.int64)
        out.fill_(-1)

        tokens = torch.empty((batch_size, max_len + 2), dtype=torch.int64)
        tokens.fill_(self.token_padding)

        for i, (tkn, proxi_matrix) in enumerate(zip(data["tokens"], data["proxi_matrix"])):
            len_seq = len(tkn) - 2 # "-2" to remove <cls> and <pad> tokens
            tokens[i, :len(tkn)] = torch.tensor(tkn)
            out[i, :len_seq, :len_seq] = proxi_matrix

        return {"name": data["name"],
                "seq": data["seq"],
                "seq_length": data["seq_length"],
                "seq_tokenized": tokens,
                "proxi_matrix": out
                }