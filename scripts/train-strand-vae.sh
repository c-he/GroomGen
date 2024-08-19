#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=1 src/train_strand_vae.py --outdir=training-runs --data=data/strand-repr/strand-frequency-repr.npz --batch=512 --lr=1e-3 --max_epochs=100 --log=1000 --snap=10 \
         --latent_dim=64 --encoder_layers=7 --decoder_layers=6 --hidden_features=1024 --strand_repr=frequency --variational=true --lambda_kl=1e-5

# torchrun --nnodes=1 --nproc_per_node=1 src/train_strand_vae.py --outdir=training-runs --data=data/strand-repr/strand-position-repr.npz --batch=512 --lr=1e-3 --max_epochs=100 --log=1000 --snap=10 \
#          --latent_dim=64 --encoder_layers=7 --decoder_layers=6 --hidden_features=1024 --strand_repr=position --variational=true --lambda_kl=1e-5