import torch
import esm

from time import time

from dataloading_utils.dataloader import get_dataloader
from simple_model import AV_Estimator


if __name__ == '__main__':
    epoch_nb = 100
    batch_size = 8 # big batch size => more padding token (see README -> The batch size conundrum)
    lr = 1e-3

    save_path = "./models"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    available_thresholds = [0.1, 0.6, 2.2, 5.1, 9.4, 14.9, 21.3, 28.4, 36, 43.9, 52.3, 61.9, 75.4, 99.3]
    threshold = 52.3
    threshold_index = available_thresholds.index(threshold)

    # backbone, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    # backbone_last_layer = 6
    backbone, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    backbone_last_layer = 12
    # backbone, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    # backbone_last_layer = 30

    backbone.eval()
    model = AV_Estimator(backbone, nb_values=1, backbone_last_layer=backbone_last_layer, device=device).to(device)

    train_dataloader, val_dataloader = get_dataloader(batch_size, alphabet, threshold_index,
                                                      use_accessibility=True, load_proxi_matrix=False,
                                                      drop_last=True,
                                                      num_workers=24)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    optim = torch.optim.Adam(model.layers.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=2, min_lr=1e-6)

    prev_best = -1
    for epoch in range(1, epoch_nb+1):
        train_loss = []
        start = time()
        for i, batch in enumerate(train_dataloader):
            input = batch["tokens"].to(device)
            target = batch["accessibility_values"].to(device)
            # target_flat = target.view(-1)

            out = model(input)  # flat because S dim merged with B dim

            loss = criterion(out.squeeze(), target)
            loss = loss[target != -100].mean()

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

                out = model(input)  # flat because S dim merged with B dim

                loss = criterion(out.squeeze(), target)
                loss = loss[target != -100].mean()
                val_loss.append(loss)

                target_flat = target.view(-1)
                out_flat = out.reshape(-1)
                out_flat = out_flat[target_flat != -100]
                target_flat = target_flat[target_flat != -100]

                out_flat[out_flat > 0] = 1.
                out_flat[out_flat <= 0] = 0.
                good_prediction = target_flat.eq(out_flat)
                top1_acc = good_prediction.sum() / target_flat.shape[0]
                top1_val.append(top1_acc.detach().cpu().data * input.shape[0])

                print(f"\rEpoch {epoch} -- VAL -- Batch {i + 1}/{len(val_dataloader)} -- Loss = {loss.data:.3f} -- Top1 = {100*top1_acc.data:.1f}", end='')
        val_top1_acc = 100 * sum(top1_val) / len(val_dataloader.dataset)
        val_avg = sum(val_loss) / len(val_loss)
        scheduler.step(val_avg)

        epoch_time = time() - start

        print(f"\nEpoch {epoch} stats :"
              f" TRAIN loss average = {train_avg:.3f}"
              f" -- VAL loss average = {val_avg:.3f}"
              f" -- ACC = {val_top1_acc:.1f}"
              f" -- time taken: {epoch_time:.0f}s")

        if val_top1_acc > prev_best:
            model_name = "threshold_" + str(threshold) + "__ACC_" + str(round(val_top1_acc.item(), 1))
            model.save_linear_layer(save_path, model_name)
            prev_best = val_top1_acc
        print()

        # if scheduler.get_last_lr()[-1] <= 1e-6:
        #     print("\nReached minimal loss => early stopping")
        #     exit()
