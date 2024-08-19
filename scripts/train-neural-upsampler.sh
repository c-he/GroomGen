#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=1 src/train_neural_upsampler.py --outdir=training-runs --data=data/neural-textures --batch=4 --lr=1e-4 --max_epochs=5 --log=1000 --snap=1 --lambda_reg=0.1