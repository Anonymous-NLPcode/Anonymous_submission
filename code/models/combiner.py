import pdb

import torch
import torch.nn as nn
# .relu(), 128--->256
class OptionSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(OptionSelector, self).__init__()

        # 神经网络层
        self.linear1 = nn.Linear(input_size,512)
        self.linear2 = nn.Linear(512, output_size)

    def forward(self, matrix):
        # # 将矩阵和向量进行扁平化
        # vec_expanded = vector.unsqueeze(1).expand(-1, 5, -1)  # 在第二维度扩展为5份
        # # 将 vec_expanded 和 matrix 拼接在一起
        # result = torch.cat((matrix, vec_expanded), dim=2)

        matrix1=self.linear1(matrix).relu()
        result=self.linear2(matrix1)
        # result = torch.matmul(matrix1.unsqueeze(0), vector1.unsqueeze(0)[0, 0])

        # 计算相似度
        # # similarities = torch.matmul(matrix, vector.T)
        # similarities=torch.bmm(vector.unsqueeze(1), matrix.permute(0,2,1))
        # weights = torch.softmax(similarities, dim=0)
        # # weights=weights.expand_as(matrix)
        # weights=weights.permute(0,2,1).expand_as(matrix)
        # # 使用权重对矩阵进行加权平均
        # result = weights * matrix
        # 经过第二个线性层得到输出
        return result

