import pdb

import torch_geometric
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import os
import ast
from gensim.models import KeyedVectors
from torch_geometric.data import HeteroData
import numpy as np
from torch_geometric.datasets import OGB_MAG
import torch.nn.functional as F
transform = T.ToUndirected()  # Add reverse edge types.

def txt2node(file_path,model_w2v):
    type_id = []  # 存储第一部分内容的列表
    type_name = []  # 存储第二部分内容的列表
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 分隔每一行的内容
            if len(parts) >= 2:  # 确保每行至少有两部分
                type_id.append(int(parts[0]))
                type_name.append(parts[1])
    type_id = torch.tensor(type_id)
    tname_vec = []
    for name in type_name:
        name = name.split('_')
        vec1 = np.mean([model_w2v[token] for token in name if token in model_w2v], axis=0)
        tname_vec.append(vec1)
    tensor_list = [torch.tensor(arr) for arr in tname_vec]
    # 将张量列表堆叠成一个3x10维的张量
    tname_vec = torch.stack(tensor_list)
    return type_id, tname_vec

def txt2edge(file_path):
    type_edgeid_1 = []  # 存储第一部分内容的列表
    type_edgeid_2 = []  # 存储第二部分内容的列表
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 分隔每一行的内容
            if len(parts) >= 2:  # 确保每行至少有两部分
                type_edgeid_1.append(int(parts[0]))
                type_edgeid_2.append(int(parts[1]))
    type_edgeid_all=[type_edgeid_1,type_edgeid_2]
    tensor_list = [torch.tensor(arr) for arr in type_edgeid_all]
    # 将张量列表堆叠
    type_edgeindex = torch.stack(tensor_list)
    return type_edgeindex

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

path='my_food/id_food.txt'
food_id,fname_vec= txt2node(path,model)
data=HeteroData()
data['food'].x=fname_vec
data['food'].y=food_id

path='my_food/id_tool.txt'
tool_id,toname_vec= txt2node(path,model)
data['tableware'].x=toname_vec

directory = "my_food"  # 替换为目标目录的路径
txt_files = [filename for filename in os.listdir(directory) if
             filename.endswith('.txt') and filename.startswith('[')]

for name in txt_files:
    path=directory+'/'+name
    data_edge=txt2edge(path)
    split_name=name[:-5][1:]
    real_name = ast.literal_eval("(" + split_name + ")")
    data[real_name].edge_index=data_edge

data['food'].train_mask=torch.ones(3, dtype=torch.bool)
data['food'].val_mask=torch.zeros(3, dtype=torch.bool)
data['food'].test_mask=torch.zeros(3, dtype=torch.bool)
# print(data.edge_index_dict)
data=transform(data)
# print(data.edge_index_dict)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

out_dict={'xdict':data.x_dict, 'edgedict':data.edge_index_dict, 'metadata':data.metadata()}
model = GNN(hidden_channels=128, out_channels=16)
model = to_hetero(model, out_dict['metadata'], aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# train_loader = HGTLoader(
#     data,
#     # Sample 15 neighbors for each node and each edge type for 2 iterations:
#     num_samples=[1024] * 4,
#     shuffle=True,
#     input_nodes=('paper', data['paper'].train_mask),
# )
with torch.no_grad():  # Initialize lazy modules.
    out = model(out_dict['xdict'], out_dict['edgedict'])

def train_1():
    model.train()
    optimizer.zero_grad()
    out = model(out_dict['xdict'], out_dict['edgedict'])
    pdb.set_trace()
    # print(data.x_dict)
    # print(data.edge_index_dict)
    mask = data['food'].train_mask
    print(out['food'][mask])
    print(data['food'].y[mask])
    # loss = F.cross_entropy(out['food'][mask], data['food'].y[mask])
    # loss.backward()
    # optimizer.step()
    return None

train_1()
