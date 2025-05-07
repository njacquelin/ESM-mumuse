import pandas as pd
import torch

from time import time


class Collate():
    def __init__(self, alphabet, use_accessibility=True, load_proxi_matrix=True):
        self.token_padding = alphabet.tok_to_idx["<pad>"]
        self.ignored_accessibility_value = -100
        self.load_proxi_matrix = load_proxi_matrix
        self.use_accessibility = use_accessibility

    def collate_fn(self, data_dict):
        """
        NOTE: it is normal that the token seq has 2 additional inputs, for cls and eos
        HOWEVER, although out and accessibility_values do not need them, we still ofset the sequence by 1 at the start
                and we add a second "empty" element in the seq at the end, in order to
                PRESERVE ALIGNMENT with the token seq
        """
        data = pd.DataFrame(data_dict)
        batch_size = len(data_dict)
        max_len = max(data['seq_length']) + 2  # +2 for cls and eos tokens

        if self.load_proxi_matrix:
            out = torch.empty((batch_size, max_len, max_len), dtype=torch.int64)
            out.fill_(-1)
        else:
            out = None

        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.token_padding)

        if self.use_accessibility:
            accessibility_values = torch.empty((batch_size, max_len), dtype=torch.float32)
            accessibility_values.fill_(self.ignored_accessibility_value)
        else:
            accessibility_values = None

        for i, (tkn, proxi_matrix, av) in enumerate(zip(data["tokens"],
                                                        data["proxi_matrix"],
                                                        data["accessibility_values"])):
            len_seq = tkn.shape[0] - 2 # "-2" to remove <cls> and <pad> tokens
            tokens[i, :tkn.shape[0]] = tkn

            # slice goes from 1 to 1+len_seq because there is the <cls> token at idx 0
            if self.load_proxi_matrix:
                out[i, 1:1+len_seq, 1:1+len_seq] = proxi_matrix
            if self.use_accessibility:
                accessibility_values[i, 1:1+len_seq] = av

        return {"name": data["name"],
                "seq": data["seq"],
                "seq_length": data["seq_length"],
                "tokens": tokens,
                "proxi_matrix": out,
                "accessibility_values": accessibility_values,
                }
