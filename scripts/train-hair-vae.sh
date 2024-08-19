#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=1 src/train_hair_vae.py --outdir=training-runs --data=data/neural-textures --batch=4 --lr=1e-3 --max_epochs=100 --log=1000 --snap=10 \
         --latent_dim=512 --variational=true --lambda_kl=1e-3