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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dnnlib
from models import StrandVAE
from torch_utils import misc, training_stats
from training.dataset import HairStrandsDataset
from training.loss import StrandVAELoss

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
@click.option('--latent_dim', help='Dimension of latent space', metavar='INT', type=click.IntRange(min=1))
@click.option('--encoder_layers', help='Number of convolutional layers for encoder', metavar='INT', type=click.IntRange(min=1))
@click.option('--decoder_layers', help='Number of convolutional layers for decoder', metavar='INT', type=click.IntRange(min=1))
@click.option('--hidden_features', help='Number of hidden units', metavar='INT', type=click.IntRange(min=1))
@click.option('--strand_repr', help='Strand representation', type=click.Choice(['position', 'frequency']), default='frequency', show_default=True)
@click.option('--variational', help='Whether to train a VAE', metavar='BOOL', default=True, show_default=True)
@click.option('--lambda_kl', help='Strength of KL divergence', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
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
        model_string_name = 'strand-vae'
        experiment_dir = f"{opts.outdir}/{experiment_index:03d}-{opts.strand_repr}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Setup data:
    dataset = HairStrandsDataset(opts.data)
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
    logger.info(f"Dataset contains {len(dataset):,} strands with {dataset.num_channels} channels ({opts.data})")

    # Create model:
    model = StrandVAE(
        in_features=dataset.num_channels,
        out_features=9 if opts.strand_repr == 'frequency' else 3,
        hidden_features=opts.hidden_features,
        latent_dim=opts.latent_dim,
        encoder_layers=opts.encoder_layers,
        decoder_layers=opts.decoder_layers,
        num_samples=51 if opts.strand_repr == 'frequency' else 99,
        variational=opts.variational,
        strand_repr=opts.strand_repr
    )
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate) and loss:
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='abs', verbose=True)
    loss = StrandVAELoss(device=device, model=model, strand_repr=opts.strand_repr, lambda_kl=opts.lambda_kl)

    # Prepare models for training:
    model.train()

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
        train_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss_val = loss.accumulate_gradients(batch=batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # important to stablize training
            optimizer.step()
            train_steps += 1
            train_loss += loss_val.item()
            # logger.info(train_steps)

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

        torch.cuda.synchronize()
        avg_loss = train_loss / len(loader)
        logger.info(f"Avg loss: {avg_loss}")
        scheduler.step(avg_loss)
        logger.info(f"Current lr: {scheduler._last_lr[0]}")
        if scheduler._last_lr[0] <= 1e-6:
            early_stop = True
        else:
            early_stop = False

        # Save checkpoint:
        if epoch % opts.snap == 0 or epoch == opts.max_epochs - 1 or early_stop:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": opts
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:04d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

        if early_stop:
            break

    model.eval()
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
