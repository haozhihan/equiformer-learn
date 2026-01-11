# run_forward_equiformer_md17_dens.py
import torch
from torch_geometric.data import Data
from nets.equiformer_md17_dens import Equiformer_MD17_DeNS

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造模型（可根据需要调整超参）
model = Equiformer_MD17_DeNS().to(device)
model.eval()  # 仅前向测试

# 构造一批简单的分子/构型数据
# 这里示例 batch_size=1，含 4 个原子
num_atoms = 4
z = torch.tensor([1, 6, 8, 1], dtype=torch.long, device=device)      # 原子序数示例: H C O H
pos = torch.randn(num_atoms, 3, device=device, requires_grad=False)  # 随机坐标
batch = torch.zeros(num_atoms, dtype=torch.long, device=device)      # 全部属于同一个分子

# 封装为 PyG Data；若有多分子，可将 batch 设为对应分子索引
data = Data(z=z, pos=pos, batch=batch)

# 前向
with torch.no_grad():  # 只想看能量/力数值时可关闭梯度；若想让力由能量梯度计算，去掉 no_grad
    energy, forces = model(data)

print("Energy shape:", energy.shape)   # 预期 [batch_size]
print("Force shape:", forces.shape)    # 预期 [num_atoms, 3]
print("Energy:", energy)
print("Forces:", forces)