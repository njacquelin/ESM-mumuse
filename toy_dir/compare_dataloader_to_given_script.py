import torch
import esm

from dataloader import get_dataloader


def representation(model, alphabet, backbone_last_layer, data):
    batch_converter = alphabet.get_batch_converter()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[backbone_last_layer], return_contacts=True)
    token_representations = results["representations"][backbone_last_layer]
    return token_representations, batch_tokens

"""
VERIFIED
(note : must be used with a debugger)
"""
if __name__ == "__main__":
    backbone, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    backbone_last_layer = 6
    device = "cuda"

    backbone = backbone.to(device)
    train_dataloader, val_dataloader = get_dataloader(2, alphabet,
                                                      use_accessibility=False, load_proxi_matrix=False,
                                                      drop_last=True,
                                                      num_workers=24)

    for batch in train_dataloader:
        batch_token_US = batch["tokens"].to(device)

        our_representation = backbone(batch_token_US, repr_layers=[backbone_last_layer])["representations"][backbone_last_layer]
        break

    data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("prot2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")]
    vanilla_representation, batch_token_THEM = representation(backbone, alphabet, backbone_last_layer, data)

    a=0




