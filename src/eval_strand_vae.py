import glob
import os

import click
import pandas as pd
import torch

from hair import load_hair, save_hair
from models import StrandVAE
from tqdm import tqdm
from utils.fft import position_to_frequency
from utils.metric import curvature
from utils.misc import copy2cpu as c2c

# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option("--indir", "-i", help="Where to load the data.", metavar="DIR", required=True)
@click.option("--outdir", "-o", help="Where to save the results.", metavar="DIR", required=True)
@click.option("--ckpt", "ckpt_path", help="Where to load pre-trained strand VAE.", metavar="DIR", required=True)
@click.option("--strand_repr", help="Strand representation", type=click.Choice(["position", "frequency"]), default="frequency", show_default=True)
@click.option("--save_strands", help="Whether to save hair strands colored by reconstruction errors.", metavar="BOOL", type=bool, default=False, show_default=True)
def eval_strand_vae(indir, outdir, ckpt_path, strand_repr, save_strands):
    hair_files = sorted(glob.glob(os.path.join(indir, "*.data")))
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if strand_repr == "position":
        model = StrandVAE(
            in_features=297,
            out_features=3,
            hidden_features=1024,
            latent_dim=64,
            encoder_layers=7,
            decoder_layers=6,
            num_samples=99,
            variational=True,
            strand_repr="position",
        )
    else:
        model = StrandVAE(
            in_features=459,
            out_features=9,
            hidden_features=1024,
            latent_dim=64,
            encoder_layers=7,
            decoder_layers=6,
            num_samples=51,
            variational=True,
            strand_repr="frequency",
        )
    model = model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)

    metrics = dict(pos_diff=[], cur_diff=[])
    for f in tqdm(hair_files):
        filename = os.path.splitext(os.path.split(f)[1])[0]
        data = torch.tensor(load_hair(f), dtype=torch.float32, device=device)
        roots = data[:, 0:1].clone()
        gt_position = data - roots
        position = torch.zeros_like(gt_position)

        max_batch = 1000
        head = 0
        with torch.no_grad():
            while head < gt_position.shape[0]:
                batch_position = gt_position[head : head + max_batch]
                if strand_repr == "position":
                    dir = batch_position[:, 1:] - batch_position[:, :-1]
                    out = model({"dir": dir})
                else:
                    amp, cos, sin = position_to_frequency(batch_position)
                    out = model({"amp": amp, "cos": cos, "sin": sin})
                position[head : head + max_batch] = out["pos"]
                head += max_batch

        pos_diff = torch.norm(gt_position - position, dim=-1)
        metrics["pos_diff"].append(c2c(pos_diff.mean()))
        cur_diff = (curvature(gt_position) - curvature(position)).abs()
        metrics["cur_diff"].append(c2c(cur_diff.mean()))

        if save_strands:
            position = position + roots
            gt_position = gt_position + roots
            save_hair(os.path.join(outdir, f"{filename}.data"), c2c(position))
            save_hair(os.path.join(outdir, f"{filename}_gt.data"), c2c(gt_position))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(os.path.join(outdir, f"{strand_repr}_metrics.csv"))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    eval_strand_vae()  # pylint: disable=no-value-for-parameter
