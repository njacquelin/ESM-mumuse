import torch
import esm
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

create_dataset = False

def custom_collate(batch):
    return batch

class CustomDataset(Dataset):
    def __init__(self, dict_data):
        # or use the RobertaTokenizer from `transformers` directly.

        self.dict_data = dict_data

    def __len__(self):
        return len(self.dict_data[list(self.dict_data.keys())[0]])

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        dict_item = {}
        for key in self.dict_data:
            if(key == "embeddings"):
                dict_item[key] = torch.tensor(self.dict_data[key][i])
            elif(key == "contact_matrix"):
                dict_item[key] = self.dict_data[key][i].tolist()
            else:
                dict_item[key] = self.dict_data[key][i]

        return dict_item


if(create_dataset or not os.path.exists("dataset.pt")):
    print("Creating torch Dataset...")
    dict_contacts = {}
    dict_proteins = {}
    max_prot_size = -1

    with open("proteinstruct/DSL_article.cm", "r") as f:
        for i, line in enumerate(f.readlines()):
            if(line[0] == "#"):
                prot_id = line.split()[3]
                if(prot_id == "contacts."): #header line copied 4 times in the file
                    continue
                dict_contacts[prot_id] = {}
            else:
                num_in_lines = line.split()
                if(int(num_in_lines[0]) not in dict_contacts[prot_id]):
                    dict_contacts[prot_id][int(num_in_lines[0])] = []
                dict_contacts[prot_id][int(num_in_lines[0])].append(int(num_in_lines[1]))


    with open("proteinstruct/DSL_article.sq", "r") as f:
        for i, line in enumerate(f.readlines()):
            if(line[0] == ">"):
                prot_id = line.split()[0].replace(">", "")
            elif(prot_id in dict_contacts):
                dict_proteins[prot_id] = line.replace("\n", "")
                if(len(line) > max_prot_size):
                    max_prot_size = len(line)

    print(f"len(dict_contacts)={len(dict_contacts)}, len(dict_proteins)={len(dict_proteins)}")
    for key in set(dict_contacts.keys())-set(dict_proteins.keys()):
        print(f"Removing {key} from dict_contact...")
        del dict_contacts[key]
    print(f"len(dict_contacts)={len(dict_contacts)}, len(dict_proteins)={len(dict_proteins)}")

    tab_prot_ids = []
    tab_proteins = []
    tab_contact_matrixes = []

    for prot_id in dict_proteins:
        tab_prot_ids.append(prot_id)
        tab_proteins.append(dict_proteins[prot_id])
        contact_matrix = np.zeros((len(dict_proteins[prot_id]), len(dict_proteins[prot_id])), dtype=np.bool)
        for i in dict_contacts[prot_id]:
            for j in dict_contacts[prot_id][i]:
                contact_matrix[i][j] = 1
        tab_contact_matrixes.append(contact_matrix)

    dataset = CustomDataset({"protein_id": tab_prot_ids, "protein_str": tab_proteins, "contact_matrix":tab_contact_matrixes})
    torch.save(dataset, "dataset.pt")
else:
    dataset = torch.load("dataset.pt", weights_only=False)




# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate)

for batch in tqdm(dataloader):
    data = [(batch[i]["protein_id"], batch[i]["protein_str"]) for i in range(len(batch))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    print(results["representations"][33].shape)
