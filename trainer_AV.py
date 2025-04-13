import torch
import esm

from time import time

from dataloader import get_dataloader
from simple_model import AV_Estimator


if __name__ == '__main__':
    epoch_nb = 10
    batch_size = 16

    save_path = "./models"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    backbone_last_layer = 12
    backbone.eval()
    model = AV_Estimator(backbone, backbone_last_layer=backbone_last_layer, device=device).to(device)

    train_dataloader, val_dataloader = get_dataloader(batch_size, alphabet, use_accessibility=True, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.layers.parameters(),
                             lr=1e-3)

    for epoch in range(epoch_nb):
        train_loss = []
        val_loss = []
        start = time()
        for i, batch in enumerate(train_dataloader):
            input = batch["tokens"].to(device)
            target = batch["accessibility_values"].to(device)

            out_flat = model(input)  # flat because S dim merged with B dim
            target_flat = target.view(-1)

            loss = criterion(out_flat, target_flat)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(f"\rEpoch {epoch} -- TRAIN -- Batch {i+1}/{len(train_dataloader)} -- Loss = {loss.data:.3f}", end='')
            train_loss.append(loss)
        print()

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input = batch["tokens"].to(device)
                target = batch["accessibility_values"].to(device)

                out_flat = model(input)
                target_flat = target.view(-1)
                loss = criterion(out_flat, target_flat)

                print(f"\rEpoch {epoch} -- VAL -- Batch {i + 1}/{len(val_dataloader)} -- Loss = {loss.data:.3f}", end='')
                val_loss.append(loss)
        print()

        train_avg = sum(train_loss) / len(train_loss)
        val_avg = sum(val_loss) / len(val_loss)
        print(f"Epoch {epoch} stats : train loss average = {train_avg:.3f} -- val loss average = {val_avg:.3f}\n")

        model.save_linear_layer(save_path, str_bonus=str(epoch) + "_epoch_" + str(val_avg) + "_loss")
