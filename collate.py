import pandas as pd
import torch

from time import time


class Collate():
    def __init__(self, alphabet):
        self.token_padding = alphabet.tok_to_idx["<pad>"]
        self.ignored_accessibility_value = -1

    def collate_fn(self, data_dict):
        data = pd.DataFrame(data_dict)
        batch_size = len(data_dict)
        max_len = max(data['seq_length'])

        out = torch.empty((batch_size, max_len, max_len), dtype=torch.int64)
        out.fill_(-1)

        tokens = torch.empty((batch_size, max_len + 2), dtype=torch.int64)  # +2 for cls and eos tokens
        tokens.fill_(self.token_padding)

        use_accessibility = not (data["accessibility_values"] is None)
        if use_accessibility:
            accessibility_values = torch.empty((batch_size, max_len), dtype=torch.int64)
            accessibility_values.fill_(self.ignored_accessibility_value)
        else:
            accessibility_values = None

        for i, (tkn, proxi_matrix, av) in enumerate(zip(data["tokens"],
                                                        data["proxi_matrix"],
                                                        data["accessibility_values"])):
            len_seq = tkn.shape[0] - 2 # "-2" to remove <cls> and <pad> tokens
            tokens[i, :tkn.shape[0]] = tkn
            out[i, :len_seq, :len_seq] = proxi_matrix
            if use_accessibility:
                accessibility_values[i, :len_seq] = av

        return {"name": data["name"],
                "seq": data["seq"],
                "seq_length": data["seq_length"],
                "seq_tokenized": tokens,
                "proxi_matrix": out,
                "accessibility_values": accessibility_values,
                }