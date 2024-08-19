"""
A minimal training script using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import logging
import os
from collections import OrderedDict
from glob import glob
from time import time

import click
import numpy as np
import PIL.Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dnnlib
from models import Discriminator, NeuralUpsampler
from torch_utils import misc, training_stats
from training.dataset import NeuralTextureDataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = 4
    gh = 2

    # Show random subset of training samples.
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, images_raw, masks, masks_raw = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(images_raw), np.stack(masks), np.stack(masks_raw)


def save_image_grid(imgs, fname, grid_size, drange=None):
    images = []
    for img in imgs:
        if drange is None:
            lo, hi = img.min(), img.max()
        else:
            lo, hi = drange
        img = (img - lo) / (hi - lo)
        images.append(img[:3])
    images = np.stack(images)
    images = np.rint(images * 255).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = images.shape
    images = images.reshape([gh, gw, C, H, W])
    images = images.transpose(0, 3, 1, 4, 2)
    images = images.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(images[..., 0], 'L').save(fname)
    else:
        PIL.Image.fromarray(images, 'RGB').save(fname)


#################################################################################
#                                  Training Loop                                #
#################################################################################
@click.command()
# Required.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--data', help='Where to load dataset.', metavar='DIR', required=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--max_epochs', help='Total training duration', metavar='INT', type=click.IntRange(min=1), required=True)
# Model hyperparameters.
@click.option('--lambda_reg', help='Strength of regularization', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
# Misc settings.
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option('--log', help='How often to print progress', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='INT', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
def main(**kwargs):
    """
    Trains a new strand-VAE.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Initialize config:
    opts = dnnlib.EasyDict(kwargs)         # Command line arguments.

    # Setup DDP:
    dist.init_process_group("nccl")
    assert opts.batch % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = opts.seed * dist.get_world_size() + rank
    batch_gpu = int(opts.batch // dist.get_world_size())
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # torch.autograd.set_detect_anomaly(True)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(opts.outdir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{opts.outdir}/*"))
        model_string_name = 'neural-upsampler'
        experiment_dir = f"{opts.outdir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        image_dir = f"{experiment_dir}/images"  # Stores saved image checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Setup data:
    dataset = NeuralTextureDataset(opts.data)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=opts.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_gpu,
        shuffle=False,
        sampler=sampler,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"Dataset contains {len(dataset):,} images with shape {dataset.raw_shape} ({opts.data})")

    # Export sample images.
    grid_size = None
    if rank == 0:
        logger.info('Exporting sample images...')
        grid_size, images, images_raw, masks, masks_raw = setup_snapshot_image_grid(training_set=dataset)
        save_image_grid(images, os.path.join(image_dir, 'reals.png'), grid_size=grid_size, drange=None)
        save_image_grid(images_raw, os.path.join(image_dir, 'reals_raw.png'), grid_size=grid_size, drange=None)
        grid_image = torch.tensor(images_raw, dtype=torch.float32, device=device).split(batch_gpu)

    # Create model:
    G = NeuralUpsampler(
        num_layers=11,
        hidden_features=128,
        img_channels=dataset.img_channels,
        raw_channels=dataset.raw_channels,
        img_resolution=dataset.img_resolution,
        raw_resolution=dataset.raw_resolution
    )
    G = DDP(G.to(device), device_ids=[rank])
    logger.info(f"Generator Parameters: {sum(p.numel() for p in G.parameters()):,}")
    D = Discriminator(
        in_features=dataset.img_channels,
        out_features=1,
        num_layers=11,
        hidden_features=128,
        batch_norm=False,
        nonlinearity='lrelu'
    )
    D = DDP(D.to(device), device_ids=[rank])
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in D.parameters()):,}")

    optimizer_G = torch.optim.Adam(G.parameters(), lr=opts.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opts.lr, betas=(0.5, 0.999))

    # Prepare models for training:
    G.train()
    D.train()

    # Initialize logs:
    if rank == 0:
        logger.info('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')

    # Variables for monitoring/logging purposes:
    train_steps = 0
    start_time = time()

    logger.info(f"Training for {opts.max_epochs} epochs...")
    for epoch in range(opts.max_epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for real_img, raw_img, _, _ in loader:
            real_img = real_img.to(device)
            raw_img = raw_img.to(device)

            ##############################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##############################################
            optimizer_D.zero_grad()

            logits_real = D(real_img).mean()
            gen_output = G(raw_img)
            logits_gen = D(gen_output['image'].detach()).mean()

            loss_D = -logits_real + logits_gen + logits_real ** 2 + logits_gen ** 2
            training_stats.report('Loss/D', loss_D)
            loss_D.backward()
            optimizer_D.step()

            ##############################################
            # (2) Update G network: maximize log(D(G(z)))
            ##############################################
            optimizer_G.zero_grad()
            gen_output = G(raw_img)
            logits_gen = D(gen_output['image']).mean()
            weight_map = gen_output['weight_map']
            loss_reg = (weight_map[:, -1] - 1).abs().mean() + weight_map[:, :4].abs().sum(dim=1).mean() + (weight_map.sum(dim=1) - 1).abs().mean()

            loss_G = -logits_gen + opts.lambda_reg * loss_reg
            training_stats.report('Loss/G', loss_G)
            loss_G.backward()
            optimizer_G.step()

            train_steps += 1

            # Log loss values:
            if train_steps % opts.log == 0:
                # torch.cuda.synchronize()
                # Print status line, accumulating the same information in training_stats.
                end_time = time()
                fields = []
                fields += [f"epoch {training_stats.report0('Progress/epoch', epoch):<5d}"]
                fields += [f"kstep {training_stats.report0('Progress/step', train_steps / 1e3):<8.1f}"]
                fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', end_time - start_time)):<12s}"]
                stats_collector.update()
                for name, value in stats_collector.as_dict().items():
                    if name.startswith("Loss/"):
                        fields += [f"{name[5:]} {value.mean:<6.3f}"]
                if rank == 0:
                    logger.info(' '.join(fields))

                # Save image snapshots:
                out = [G(img) for img in grid_image]
                images = torch.cat([o['image'].detach().cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(image_dir, f'fakes{train_steps:06d}.png'), grid_size=grid_size, drange=None)

        # Save checkpoint:
        if epoch % opts.snap == 0 or epoch == opts.max_epochs - 1:
            if rank == 0:
                checkpoint = {
                    "G": G.module.state_dict(),
                    "opt_G": optimizer_G.state_dict(),
                    "D": D.module.state_dict(),
                    "opt_D": optimizer_D.state_dict(),
                    "args": opts
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:04d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
