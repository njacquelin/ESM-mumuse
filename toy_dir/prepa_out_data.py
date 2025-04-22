import pandas as pd
from linecache import getline, getlines
import pickle
import torch

from time import time

# def get_proxi_matrix(prot, path_out):
#     local_prot = None
#     next_line = 2
#     while prot != local_prot:
#         current_line = next_line
#         head_line = getline(path_out, current_line)
#         head_data = head_line.split(" ")
#         seq_len = int(head_data[1])
#         nb_of_ones = int(head_data[2])
#         local_prot = head_data[3]
#         if local_prot == '7ODCA' and current_line != 2:
#             print(f"PROBLEM: reached end of quarter file wothout finding {prot} => closing now")
#             exit()
#         next_line += int(nb_of_ones) + 1
#     data = torch.zeros((seq_len, seq_len))
#     for i in range(nb_of_ones):
#         x_y = getline(path_out, current_line + i + 1).split(" ")
#         data[int(x_y[0]), int(x_y[1])] = 1
#     return data

def get_proxi_matrix_faster(metadata, prot, path_out):
    prot_metadata = metadata[prot]
    seq_len = prot_metadata['seq_len']
    data = torch.zeros((seq_len, seq_len))
    for i in range(prot_metadata['nb_of_ones']):
        x_y = getline(path_out, prot_metadata['current_line'] + i + 1).split(" ")
        data[int(x_y[0]), int(x_y[1])] = 1
    return data


def get_metadata(path_out, path_lst):
    next_line = 2
    metadata = {}
    while next_line != 1544180:
        current_line = next_line
        head_line = getline(path_out, current_line)
        head_data = head_line.split(" ")
        seq_len = int(head_data[1])
        nb_of_ones = int(head_data[2])
        local_prot = head_data[3]
        next_line += int(nb_of_ones) + 1
        infos = {'prot': local_prot,
                 'seq_len': seq_len,
                 'nb_of_ones': nb_of_ones,
                 'current_line': current_line}
        metadata[local_prot] = infos

        with open(path_lst, 'r') as file:
            prot_list = file.read()
        prot_list = prot_list.split('\n')
    return metadata, prot_list


def getitem(i, metadata, prot_list, path_sq, path_out):
    prot = prot_list[i]
    data = {"name": prot,
            "index": i}
    data["seq"] = getline(path_sq, i * 2 + 2)
    data["proxi_matrix"] = get_proxi_matrix_faster(metadata, prot, path_out)
    return data


path_sq = "../data/quarter_cleaned.sq.txt"
metadata, prot_list = get_metadata(path_out="../data/DSL_article.cm", path_lst="../data/DSL_article.lst")
path_out = "../data/DSL_article.cm"

start = time()
data = getitem(3330, metadata, prot_list, path_sq, path_out)
end = time()
print(end - start)
a=0





# if __name__=='__main__':
#     path = "./data/"
#     path_av = path + "quarter_cleaned.av.txt"
#     path_sq = path + "quarter_cleaned.sq.txt"
#     path_lst = path + "DSL_article.lst"
#     path_out = path + "DSL_article.cm"
#
#     with open(path_lst, 'r') as file:
#         prot_list = file.read()
#     prot_list = prot_list.split('\n')
#
#     start = time()
#     data = getitem(3330, prot_list, path_sq, path_out)
#     end = time()
#     print(end - start)
#     a=0
#
#     # all_data = pd.DataFrame()
#     # for i, prot in tqdm(enumerate(prot_list[:-1])):
#
#         # # with open('filename.pickle', 'wb') as handle:
#         # #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         #
#         # # assignation, really ?
#         # all_data = pd.concat([all_data, pd.DataFrame([data])], ignore_index=True)
#         # if i%500 == 0: print(i)
