#!/bin/bash

# python src/eval_strand_vae.py --indir=data/public-hair --outdir=evaluation/strand-vae --ckpt=pretrained-models/strand-vae/position-strand-vae.pt --strand_repr=position --save_strands=true
# python src/eval_strand_vae.py --indir=data/public-hair --outdir=evaluation/fft-strand-vae --ckpt=pretrained-models/strand-vae/frequency-strand-vae.pt --strand_repr=frequency --save_strands=true
python src/eval_strand_vae.py --indir=data/test --outdir=evaluation/strand-vae --ckpt=pretrained-models/strand-vae/position-strand-vae.pt --strand_repr=position --save_strands=true
python src/eval_strand_vae.py --indir=data/test --outdir=evaluation/fft-strand-vae --ckpt=pretrained-models/strand-vae/frequency-strand-vae.pt --strand_repr=frequency --save_strands=true