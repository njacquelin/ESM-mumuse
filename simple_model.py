import torch
from torch import nn

import os

from itertools import product


class AV_Estimator(nn.Module):
    def __init__(self,
                 backbone,
                 backbone_last_layer,
                 device,
                 h_size=256,
                 nb_values=1):
        super(AV_Estimator, self).__init__()

        self.backbone = backbone.to(device)
        self.backbone_last_layer = backbone_last_layer

        self.input_size = backbone.embed_dim
        self.h_size = h_size
        self.nb_values = nb_values


        # ~23% Top1 acc with esm2_t30_150M_UR50D backbone
        # ~19% Top1 acc with esm2_t6_8M_UR50D backbone
        # self.layers = nn.Sequential(
        #     nn.Linear(self.input_size, self.h_size, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.h_size),
        #     nn.Linear(self.h_size, self.nb_values, bias=False),
        #     # nn.Softmax(dim=1)
        # )

        self.layers = nn.Sequential(
            nn.Conv1d(self.input_size, self.h_size, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.h_size),
            nn.Conv1d(self.h_size, self.h_size, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.h_size),
            nn.Conv1d(self.h_size, self.nb_values, kernel_size=1, padding=0, bias=False),
            ## nn.Sigmoid() <= NOPE, because we use the (more stable) BCE with logits loss
        )

    def forward(self, x):
        with torch.no_grad():
            emb = self.backbone(x, repr_layers=[self.backbone_last_layer])["representations"][self.backbone_last_layer]
        # emb_no_padding = emb[x != 1, :]
        # flat_emb = emb_no_padding.reshape(-1, self.input_size)
        emb = emb.swapaxes(1, 2)
        out = self.layers(emb)
        return out

    def save_linear_layer(self, path, model_name=""):
        model_path = os.path.join(path, model_name + ".pt")
        torch.save(self.layers.state_dict(), model_path)
        print("\tSAVED at:", model_path)

    def load_linear_layer(self, path):
        self.layers.load_state_dict(torch.load(path, weights_only=True))
        print("Linear layer loaded.")


class Proximity_Detector_with_Cats(AV_Estimator):
    def __init__(self):
        super(Proximity_Detector_with_Cats, self).__init__()

        self.norm = 1 / self.h_size

        self.mask = {}

        self.proxi_estimator = nn.Sequential(
            nn.Linear(self.input_size, self.h_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.h_size),
            nn.Linear(self.h_size, 1),
            # nn.Sigmoid() # <== removed so that we use the BCE_logsoftmax loss for more stable results
        )

    # Not tested yet => TODO: test it
    def forward(self, x):
        with torch.no_grad():
            emb = self.backbone(x, repr_layers=[self.backbone_last_layer])["representations"][self.backbone_last_layer]
        out_flatten = self.layers(emb)
        out = out_flatten.view(x.shape[0], -1, self.nb_values)

        richer_emb = torch.cat((emb, out), dim=2)

        cat_square_matrix = self.fast_cat_square(richer_emb, richer_emb)
        cat_matrix = self.layers(cat_square_matrix)
        return cat_matrix

    # from: https://stackoverflow.com/questions/69027228/all-possible-concatenations-of-two-tensors-in-pytorch
    def fast_cat_square(self, S, T):
        """
        concatenates each vector in the batch of S with eah vector in the batch of T
        returns a matrix of shape: (B, B, 2*D)
        """
        batch_size = S.shape[0]
        if batch_size not in self.mask.keys():
            self.mask[batch_size] = self.get_mask(batch_size)
        ST = torch.stack((S, T)).reshape(batch_size*2, self.input_size)
        catted_seq = self.mask[batch_size] @ ST
        cat_matrix = catted_seq.reshape(batch_size, batch_size, self.input_size*2)
        return cat_matrix

    def get_mask(self, batch_size):
        indices = torch.tensor(list(product(range(0, batch_size), range(batch_size, batch_size * 2))))
        mask = nn.functional.one_hot(indices).float()
        return mask

