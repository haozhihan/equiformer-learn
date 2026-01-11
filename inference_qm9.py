"""
简洁的 QM9 Equiformer 推理脚本
用于加载 checkpoint 并进行推理
"""
import torch
import argparse
from torch_geometric.loader import DataLoader

from datasets.pyg.qm9 import QM9
from nets import model_entrypoint


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """加载 checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理 DDP 模型的 state_dict (移除 'module.' 前缀)
    state_dict = checkpoint['state_dict']
    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ 成功加载 checkpoint (epoch: {checkpoint.get('epoch', 'N/A')})")
    if 'best_val_err' in checkpoint:
        print(f"  Best val MAE: {checkpoint['best_val_err']:.5f}")
        print(f"  Best test MAE: {checkpoint['best_test_err']:.5f}")
    
    return model


def inference(model, data_loader, target, device, task_mean=0, task_std=1):
    """在数据集上进行推理"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # 模型前向传播
            pred = model(
                f_in=data.x,
                pos=data.pos,
                batch=data.batch,
                node_atom=data.z,
                edge_d_index=data.edge_d_index,
                edge_d_attr=data.edge_d_attr
            )
            pred = pred.squeeze()
            
            # 反标准化 (如果需要)
            pred = pred * task_std + task_mean
            true = data.y[:, target]
            
            predictions.append(pred.cpu())
            targets.append(true.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    # 计算 MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    return predictions, targets, mae


def main():
    parser = argparse.ArgumentParser('QM9 Equiformer 推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint 路径')
    parser.add_argument('--data-path', type=str, default='datasets/qm9',
                        help='数据集路径')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'valid', 'test'],
                        help='使用哪个数据集分割')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    # 模型参数 (需要与训练时一致)
    parser.add_argument('--model-name', type=str, 
                        default='graph_attention_transformer_nonlinear_bessel_l2_drop00',
                        help='模型名称')
    parser.add_argument('--input-irreps', type=str, default='5x0e',
                        help='输入不可约表示')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='截断半径')
    parser.add_argument('--num-basis', type=int, default=8,
                        help='径向基函数数量')
    parser.add_argument('--target', type=int, default=7,
                        help='预测目标 (QM9 属性索引)')
    parser.add_argument('--feature-type', type=str, default='one_hot',
                        help='特征类型')
    parser.add_argument('--task-mean', type=float, default=0.0,
                        help='任务均值 (如果使用了标准化)')
    parser.add_argument('--task-std', type=float, default=1.0,
                        help='任务标准差 (如果使用了标准化)')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"\n加载 {args.split} 数据集...")
    dataset = QM9(args.data_path, args.split, feature_type=args.feature_type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"数据集大小: {len(dataset)}")
    
    # 创建模型
    print(f"\n创建模型: {args.model_name}")
    create_model = model_entrypoint(args.model_name)
    model = create_model(
        irreps_in=args.input_irreps,
        radius=args.radius,
        num_basis=args.num_basis,
        out_channels=1,
        task_mean=args.task_mean,
        task_std=args.task_std,
        atomref=None,
        drop_path=0.0
    )
    model = model.to(device)
    
    # 加载 checkpoint
    print(f"\n加载 checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, model, device)
    
    # 推理
    print(f"\n开始推理...")
    predictions, targets, mae = inference(
        model, data_loader, args.target, device,
        task_mean=args.task_mean, task_std=args.task_std
    )
    
    print(f"\n{'='*50}")
    print(f"推理结果:")
    print(f"  MAE: {mae:.5f}")
    print(f"  预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  真实值范围: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
