import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module import Generator


class NeuralUpsampler(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_features,
        img_channels,
        raw_channels,
        img_resolution,
        raw_resolution,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.img_channels = img_channels
        self.raw_channels = raw_channels
        self.img_resolution = img_resolution
        self.raw_resolution = raw_resolution
        in_features = (
            4 + 1
        ) * raw_channels + 4  # concatenation of its 4 neighboring guides, their bilinear interpolation, and the distances to the guides
        self.generator = Generator(
            in_features=in_features,
            out_features=5,
            num_layers=num_layers,
            hidden_features=hidden_features,
            instance_norm=True,
            nonlinearity="relu",
        )

        # KNN index required for blending.
        u, v = torch.meshgrid(
            torch.linspace(0, 1, steps=self.img_resolution),
            torch.linspace(0, 1, steps=self.img_resolution),
            indexing="ij",
        )
        uv = torch.dstack((u, v)).permute(2, 1, 0)
        uv_guide = F.interpolate(
            uv.unsqueeze(0),
            size=(self.raw_resolution, self.raw_resolution),
            mode="nearest",
        )[0]

        uv = uv.permute(1, 2, 0).reshape(-1, 2)
        uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
        dist = torch.norm(uv.unsqueeze(1) - uv_guide.unsqueeze(0), dim=-1)
        knn_dist, knn_index = dist.topk(4, largest=False)
        self.register_buffer("knn_index", knn_index.flatten())
        self.register_buffer(
            "knn_dist", knn_dist.T.reshape(4, self.img_resolution, self.img_resolution)
        )

    def forward(self, img):
        N = img.shape[0]
        upsampled = F.interpolate(
            img,
            (self.img_resolution, self.img_resolution),
            mode="bilinear",
            align_corners=False,
        )
        guides = img.reshape(N, self.img_channels, -1)
        guides = guides.index_select(dim=-1, index=self.knn_index)
        guides = guides.reshape(
            N, self.img_channels, self.img_resolution, self.img_resolution, 4
        )
        guides_x = guides.permute(0, 1, 4, 2, 3).reshape(
            N, -1, self.img_resolution, self.img_resolution
        )
        dists = self.knn_dist.unsqueeze(0).repeat(N, 1, 1, 1)
        x = torch.cat(
            [
                guides_x[:, : self.raw_channels],
                upsampled[:, : self.raw_channels],
                dists,
            ],
            dim=1,
        )
        weight_map = self.generator(x)
        out = (
            torch.einsum("nchwx,nxhw->nchw", guides, weight_map[:, :4])
            + upsampled * weight_map[:, -1:]
        )
        return {"image": out, "weight_map": weight_map}
