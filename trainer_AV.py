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

    train_dataloader, val_dataloader = get_dataloader(batch_size, alphabet,
                                                      use_accessibility=True, load_proxi_matrix=False,
                                                      drop_last=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optim = torch.optim.Adam(model.layers.parameters(),
                             lr=1e-3)

    for epoch in range(1, epoch_nb+1):
        train_loss = []
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
        train_avg = sum(train_loss) / len(train_loss)
        print()

        val_loss = []
        top1_val = []
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input = batch["tokens"].to(device)
                target = batch["accessibility_values"].to(device)

                out_flat = model(input)
                target_flat = target.view(-1)

                loss = criterion(out_flat, target_flat)
                val_loss.append(loss)

                _, top1 = out_flat.topk(k=1, dim=1)
                good_prediction = target_flat[target_flat != -100].eq(top1[:, 0][target_flat != -100])
                top1_acc = good_prediction.sum() / good_prediction.shape[0]
                top1_val.append(top1_acc.detach().cpu().data * input.shape[0])

                print(f"\rEpoch {epoch} -- VAL -- Batch {i + 1}/{len(val_dataloader)} -- Loss = {loss.data:.3f} -- Top1 = {top1_acc.data}", end='')
        val_top1_acc = 100 * sum(top1_val) / len(val_dataloader.dataset)
        print()

        val_avg = sum(val_loss) / len(val_loss)
        print(f"Epoch {epoch} stats : train loss average = {train_avg:.3f} "
              f"-- val loss average = {val_avg:.3f} "
              f"-- val top1 = {val_top1_acc:.1f}\n")

        model.save_linear_layer(save_path, str_bonus=str(epoch) + "_epoch_" + str(val_avg) + "_loss")
