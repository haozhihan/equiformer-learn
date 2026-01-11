#!/bin/bash
# QM9 Equiformer 推理脚本使用示例

# 基本用法 - 使用默认参数
python inference_qm9.py \
    --checkpoint 'models/qm9/equiformer/se_l2/target@7/lr@1.5e-4_epochs@10_bs@64_wd@0.0_dropout@0.0_bessel@8_no-stad_l1-loss_g@4/best_checkpoint.pth.tar' \
    --split test

# 完整参数示例
python inference_qm9.py \
    --checkpoint 'models/qm9/equiformer/se_l2/target@7/lr@1.5e-4_epochs@10_bs@64_wd@0.0_dropout@0.0_bessel@8_no-stad_l1-loss_g@4/best_checkpoint.pth.tar' \
    --data-path 'datasets/qm9' \
    --split test \
    --batch-size 64 \
    --device cuda \
    --model-name 'graph_attention_transformer_nonlinear_bessel_l2_drop00' \
    --input-irreps '5x0e' \
    --radius 5.0 \
    --num-basis 8 \
    --target 7 \
    --feature-type 'one_hot' \
    --task-mean 0.0 \
    --task-std 1.0
