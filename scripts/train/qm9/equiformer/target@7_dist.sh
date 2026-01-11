#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

# export PYTHONNOUSERSITE=True    # prevent using packages from base
# conda activate equiformer

python -m torch.distributed.launch --nproc_per_node=4 --use_env main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@7/lr@1.5e-4_epochs@10_bs@64_wd@0.0_dropout@0.0_bessel@8_no-stad_l1-loss_g@4' \
    --model-name 'graph_attention_transformer_nonlinear_bessel_l2_drop00' \
    --input-irreps '5x0e' \
    --target 7 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --batch-size 32 \
    --radius 5.0 \
    --num-basis 8 \
    --drop-path 0.0 \
    --weight-decay 0.0 \
    --lr 1.5e-4 \
    --epochs 10 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp \
    --no-standardize
