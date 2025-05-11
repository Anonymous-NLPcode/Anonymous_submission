import pdb
import torch
import torch_geometric.transforms as T
import os
import ast
from gensim.models import KeyedVectors
from torch_geometric.data import HeteroData
import numpy as np

package_dir = os.path.dirname(os.path.abspath(__file__))
bin_file = os.path.join(package_dir, 'GoogleNews-vectors-negative300.bin')
model_w2v = KeyedVectors.load_word2vec_format(bin_file, binary=True)

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
    # tensor_list = [torch.tensor(arr) for arr in tname_vec]
    tensor_list=[]
    for arr in tname_vec:
        if np.isnan(np.sum(arr)):
            arr=torch.tensor(np.mean([model_w2v['None']],axis=0))
        else:
            arr=torch.tensor(arr)
        tensor_list.append(arr)

    # 将张量列表堆叠
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

def get_lis_data(fi_path):

    path = fi_path + 'id_food.txt'
    food_id, fname_vec = txt2node(path, model_w2v)
    data = HeteroData()
    data['food'].x = fname_vec
    data['food'].y = food_id

    path = fi_path + 'id_tool.txt'
    tool_id, toname_vec = txt2node(path, model_w2v)
    data['tableware'].x = toname_vec

    txt_files = [filename for filename in os.listdir(fi_path[:-1]) if
                 filename.endswith('.txt') and filename.startswith('[')]

    for name in txt_files:
        path = fi_path + name
        data_edge = txt2edge(path)
        split_name = name[:-5][1:]
        real_name = ast.literal_eval("(" + split_name + ")")
        data[real_name].edge_index = data_edge
    transform = T.ToUndirected()  # Add reverse edge types.
    data = transform(data)
    data.cuda()
    return data

# a=get_lis_data('my_food_graph/001.my_food/001/')
# print(a.metadata()[1])
