import torch
import esm
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from dataloading_utils.dataloader import get_dataloader
from simple_model import AV_Estimator


def get_confusion_matrix(model, val_dataloader):
    cm = np.zeros((14, 14))
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input = batch["tokens"].to(device)
            target = batch["accessibility_values"]

            target_flat = target.view(-1).numpy()
            out = model(input)

            out_flat = out.swapaxes(1, 2).reshape(-1, 14)

            _, top1 = out_flat.topk(k=1, dim=1)
            top1 = top1[:, 0].cpu().numpy()
            top1 = top1[target_flat != -100]
            target_flat = target_flat[target_flat != -100]

            cm += confusion_matrix(target_flat, top1)
    cm /= cm.sum(axis=0)

    plt.imshow(cm)
    plt.show()
    return cm


def get_top_N_acc(model, dataloader, n=1):
    top = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input = batch["tokens"].to(device)
            target = batch["accessibility_values"]

            out = model(input)

            target_flat = target.view(-1).to(device)
            out_flat = out.swapaxes(1, 2).reshape(-1, 14)
            out_flat = out_flat[target_flat != -100]
            target_flat = target_flat[target_flat != -100]

            _, pred = out_flat.topk(k=n, dim=1)
            correct = target_flat.unsqueeze(1).eq(pred).float()
            top += correct.sum() / target_flat.shape[0]
    top /= len(dataloader)
    return float(top * 100)


def get_BINARY_confusion_matrix(model, val_dataloader):
    cm = np.zeros((2, 2))
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input = batch["tokens"].to(device)
            target = batch["accessibility_values"]

            target_flat = target.view(-1).numpy()
            out = model(input).reshape(-1)

            out = out[target_flat != -100]
            target_flat = target_flat[target_flat != -100]

            out[out > 0] = 1.
            out[out <= 0] = 0.
            top1 = out.cpu().numpy()

            cm += confusion_matrix(target_flat, top1)
    cm /= cm.sum(axis=0)

    plt.imshow(cm)
    plt.show()
    return cm



if __name__ == '__main__':
    batch_size = 8
    threshold = 5.1

    load_path = "./models/threshold_9.4__ACC_78.3.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    backbone_last_layer = 6
    # backbone, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    # backbone_last_layer = 12
    # backbone, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    # backbone_last_layer = 30

    model = AV_Estimator(backbone, nb_values=1, backbone_last_layer=backbone_last_layer, device=device).to(device)
    model.load_linear_layer(load_path)

    _, val_dataloader = get_dataloader(batch_size, alphabet, threshold,
                                       use_accessibility=True, load_proxi_matrix=False,
                                       drop_last=True)

    cm = get_BINARY_confusion_matrix(model, val_dataloader)
    per_class_acc = 100 * np.trace(cm) / cm.shape[0]
    print(f"Per Class Accuracy: {per_class_acc:.1f}")


