from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd

from linecache import getline
from dataloading_utils.collate import Collate


from time import time


class Prot_Dataset(Dataset):
    def __init__(self, alphabet,
                 use_accessibility=False,
                 load_proxi_matrix=True,
                 path="./data/",
                 file_av="quarter_cleaned.av.txt",
                 file_sq="quarter_cleaned.sq.txt",
                 file_lst="DSL_article.lst",
                 path_out="DSL_article.cm"):
        # data access stuff
        self.path_av = path + file_av
        self.path_sq = path + file_sq
        self.path_lst = path + file_lst
        self.path_out = path + path_out
        self.load_proxi_matrix = load_proxi_matrix
        self.use_accessibility = use_accessibility
        self.metadata = {}
        self.init_metadata()
        self.prot_list = list(self.metadata.keys())
        self.len = len(self.prot_list)

        # tokenizetion stuff
        self.batch_converter = alphabet.get_batch_converter()
        self.token_start = [alphabet.tok_to_idx["<cls>"]]
        self.token_padding = alphabet.tok_to_idx["<pad>"]
        self.token_oes = [alphabet.tok_to_idx["<eos>"]]
        self.conversion_fn = alphabet.encode

    def str_to_tokens(self, seq):
        return torch.tensor(self.token_start + self.conversion_fn(seq) + self.token_oes)

    def init_metadata(self):
        next_line = 2
        iteration = 0
        while next_line != 1544180: # as the data repeat 4 times in the data file, we stop after the first quarter
            current_line = next_line
            head_line = getline(self.path_out, current_line)
            head_data = head_line.split(" ")
            seq_len = int(head_data[1])
            nb_of_ones = int(head_data[2])
            local_prot = head_data[3]

            if self.use_accessibility:
                av_name_line = 6 + iteration * 2
                assert local_prot in getline(self.path_av, av_name_line)
                    # print(f"Problem with av: data for {local_prot} is not at line {av_name_line}.")
                    # exit()
                av_line_data = av_name_line + 1
                iteration += 1
            else:
                av_line_data = -1

            next_line += int(nb_of_ones) + 1
            infos = {'prot': local_prot,
                     'seq_len': seq_len,
                     'nb_of_ones': nb_of_ones,
                     'current_line': current_line,
                     'av_line_data': av_line_data}
            if local_prot in self.metadata.keys(): break # just to be sure we don't go over the data for a second time
            self.metadata[local_prot] = infos

    def get_proxi_matrix_faster(self, prot, symmetry=True):
        prot_metadata = self.metadata[prot]
        seq_len = prot_metadata['seq_len']
        data = torch.zeros((seq_len, seq_len))
        for i in range(prot_metadata['nb_of_ones']):
            x_y = getline(self.path_out, prot_metadata['current_line'] + i + 1).split(" ")
            data[int(x_y[0]), int(x_y[1])] = 1
            if symmetry: data[int(x_y[1]), int(x_y[0])] = 1
        return data, seq_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        prot = self.prot_list[idx]
        if self.load_proxi_matrix:
            proxi_matrix, out_length = self.get_proxi_matrix_faster(prot)
        else:
            proxi_matrix, out_length = None, None
        seq = getline(self.path_sq, idx * 2 + 2)[:-1] # [:-1] to remove the '\n' character
        seq_length = len(seq)
        tokens = self.str_to_tokens(seq)
        accesibility_values = self.get_accesibility_values(prot) if self.use_accessibility else None
        return {"name": prot,
                "index": idx,
                "out_length": out_length,
                "seq": seq,
                "tokens": tokens,
                "seq_length": seq_length,
                "proxi_matrix": proxi_matrix,
                "accessibility_values": accesibility_values,
               }

    def get_accesibility_values(self, prot):
        av_line = self.metadata[prot]["av_line_data"]
        accesibility_values_str = getline(self.path_av, av_line)
        accesibility_values_str = list(accesibility_values_str.split(" ")[:-1])  # "-1" is "\n" at the end of the line
        accesibility_values_int = list(map(int, accesibility_values_str))
        accesibility_values = torch.tensor(accesibility_values_int)
        return accesibility_values


def get_dataloader(batch_size, alphabet, use_accessibility, load_proxi_matrix, drop_last, num_workers=0, train_split=0.8):
    collator = Collate(alphabet, use_accessibility=use_accessibility, load_proxi_matrix=load_proxi_matrix).collate_fn

    full_dataset = Prot_Dataset(alphabet, use_accessibility=use_accessibility, load_proxi_matrix=load_proxi_matrix)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator,
                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator,
                                shuffle=True, num_workers=num_workers, drop_last=drop_last)
    return train_dataloader, val_dataloader



#### TEST STUFF => COMMENT ME IF FINISHED ! #####
# import esm
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model.eval()  # disables dropout for deterministic results
# t, v = get_dataloader(2, alphabet, True)
# # t.dataset.dataset.__getitem__(0)
# # a=0
# for b in t:
#     a=0
#################################################
