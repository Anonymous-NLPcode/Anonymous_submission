import torch

# 原始数据
data = torch.tensor([[-0.0914, -0.0455, -0.0455, -0.0354, -0.1544, -0.0339],
                    [0.1731, 0.1709, 0.1527, 0.1691, 0.1560, 0.1305],
                    [-0.0776, -0.0773, -0.0904, -0.0867, -0.1179, -0.0969],
                    [0.1452, 0.1245, 0.1269, 0.1269, 0.1557, 0.0371]], device='cuda:0')

# 找到每行的最大值的索引, 将每行最大值所在位置置为1
max_indices = torch.argmax(data, dim=1)
result = torch.zeros_like(data)
result[torch.arange(data.shape[0]), max_indices] = 1

print(result)
