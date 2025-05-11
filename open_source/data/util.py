import pdb
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import hausdorff
import numba
import itertools
import pandas as pd
import math
import torch.nn.functional as F

def stack_pos_neg(pos_imgs, neg_imgs):
    """
    Given positive images and negative images, stack them, then create a y tensor of labels.

    TODO - instead of stack just assign?
    """
    pos_imgs = torch.stack(pos_imgs)
    neg_imgs = torch.stack(neg_imgs)

    imgs = torch.cat([pos_imgs, neg_imgs], 0)
    y = torch.zeros(imgs.shape[0], dtype=torch.uint8)
    y[: pos_imgs.shape[0]] = 1
    return imgs, y

def pad_tensor_to_length(tensor, target_length, pad_value=0):
    current_length = tensor.size(0)
    if current_length >= target_length:
        return tensor
    else:
        padding_size = target_length - current_length
        padding = torch.full((padding_size,) + tensor.size()[1:], pad_value, dtype=tensor.dtype)
        return torch.cat((tensor, padding))

def split_spk_lis(inp, y, n_examples, names, tr_ratio, percent_novel=0.0):

    # n_pos_ex = n_examples // 2        # 3
    n_pos_ex = 1
    spk_inp = torch.zeros((n_examples, *inp.shape[1:]), dtype=inp.dtype)
    spk_inp[:n_pos_ex] = inp[:n_pos_ex]
    # spk_inp[n_pos_ex:] = inp[midp : midp + n_pos_ex]
    spk_inp[n_pos_ex:] = inp[n_pos_ex: n_examples]
    a=[]
    a.append(names[0])
    names[0]=a
    spk_label = torch.zeros(n_examples, dtype=torch.uint8)
    # spk_label[torch.randint(0, 5, (1,))] = 1
    spk_label[0] = 1
    lis_inp = []
    lis_label = torch.zeros(n_examples, dtype=torch.uint8)
    if percent_novel == 0.0:
        lis_label = spk_label
        lis_inp = spk_inp
    boexs_name=[]
    for n in names:
        for nam in n:
            b = nam[:7] + '_boxes' + nam[7:]
            b = b[:-4] + '/metadata.csv'
            b = 'data/' + b
            boexs_name.append(b)

    num_fea=10
    all_boxes_data=[]
    for bn in boexs_name:
        boex_data = pd.read_csv(bn)
        if len(boex_data) <= 10:
            while len(boex_data) <= 10:
                avg_values = boex_data.iloc[:len(boex_data)].mean()
                # 在 DataFrame 底部添加一行
                boex_data = boex_data.append(avg_values, ignore_index=True)
        top_data = boex_data.nlargest(num_fea, 'area')
        top_rows = top_data.values.tolist()
        all_boexs=[]
        for sigbox in top_rows:
            sigbox = sigbox[2:6]
            all_boexs.append(sigbox)
        all_boxes_data.append(all_boexs)

    for i in range(0, len(all_boxes_data)):
        box = all_boxes_data[i]
        for n in range(0,len(box)):
            box[n][0] = box[n][0] * tr_ratio[i][1]
            box[n][1] = box[n][1] * tr_ratio[i][0]
            box[n][2] = box[n][2] * tr_ratio[i][1]
            box[n][3] = box[n][3] * tr_ratio[i][0]
        all_boxes_data[i] = box
    all_imgs_part = []
    all_center_cord = []
    all_edges = []
    for i in range(0, len(all_boxes_data)):
        img_part=[]
        center_cord = []
        img = spk_inp[i]
        box = all_boxes_data[i]
        for n in range(0, len(box)):
            x_min = round(box[n][0])
            x_max = round(box[n][0] + box[n][2])
            y_min = round(box[n][1])
            y_max = round(box[n][1] + box[n][3])
            center_cord.append(((y_min+y_max)/2, (x_min+x_max)/2))
            desired_size = (224, 224)  # 你的CNN网络期望的输入尺寸
            box_img = img[:, y_min:y_max, x_min:x_max]
            # box_img[0] = torch.tensor(cv2.resize(np.array(box_img[0]), desired_size))
            try:
                box_img = F.interpolate(box_img.unsqueeze(0), size=desired_size, mode='nearest').squeeze(0)
            except RuntimeError as e:
                pdb.set_trace()
            img_part.append(box_img)   # N个
        all_imgs_part.append(img_part)        # 5 * 10 * (3 ,244, 244)
        all_center_cord.append(center_cord)     # 5 * 10 * (x, y)

    N = 3
    for obj_cord in all_center_cord:
        closest_points = []
        for ii in range(len(obj_cord)):
            current_point = obj_cord[ii]
            obj_distances = []
            for j in range(len(obj_cord)):
                if ii != j:
                    distance = math.sqrt(
                        (current_point[0] - obj_cord[j][0]) ** 2 + (current_point[1] - obj_cord[j][1]) ** 2)
                    obj_distances.append(((ii, j), distance))
            obj_distances.sort(key=lambda x: x[1])
            obj_items = [item[0] for item in obj_distances]
            closest_points.append((obj_items[:N]))
        # 将数据扁平化并转换为两个列表
        flattened_data = [[coord for pair in group for coord in pair] for group in closest_points]
        # 使用 zip(*flattened_data) 将列表转置
        result = [list(group) for group in zip(*flattened_data)]
        # 串联双数索引的子列表
        even_index_lists = [item for i, item in enumerate(result) if i % 2 == 1]
        concatenated_even = [element for sublist in even_index_lists for element in sublist]
        # 串联单数索引的子列表
        odd_index_lists = [item for i, item in enumerate(result) if i % 2 == 0]
        concatenated_odd = [element for sublist in odd_index_lists for element in sublist]
        # 将串联得到的两个列表放到一个列表里
        result = [concatenated_odd, concatenated_even]
        edge = torch.tensor(np.array(result))
        all_edges.append(edge)
    all_imgs_part = torch.stack([torch.stack(sublist, dim=0) for sublist in all_imgs_part], dim=0)
    return spk_inp, spk_label, lis_inp, lis_label, names, all_imgs_part, all_edges
def train_test_split_pt(*tensors, test_size=0.2):
    """
    Pytorch train test split
    """
    first = tensors[0]
    if isinstance(test_size, float):
        test_size = int(first.shape[1] * test_size)
    perm = torch.randperm(first.shape[1])

    train_perm = []
    test_perm = []
    for t in tensors:
        t_perm = t[:, perm]
        train_perm.append(t_perm[:, test_size:].contiguous())
        test_perm.append(t_perm[:, :test_size].contiguous())

    return train_perm, test_perm


def train_val_test_split(data, val_size=0.1, test_size=0.1, random_state=None):
    """
    Split data into train, validation, and test splits
    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    """
    idx = np.arange(data["imgs"].shape[0])
    idx_train, idx_valtest = train_test_split(
        idx, test_size=val_size + test_size, random_state=random_state, shuffle=True
    )
    idx_val, idx_test = train_test_split(
        idx_valtest,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        shuffle=True,
    )
    splits = []
    for idx_split in (idx_train, idx_val, idx_test):
        splits.append(
            {
                "imgs": data["imgs"][idx_split],
                "labels": data["labels"][idx_split],
                "langs": data["langs"][idx_split],
            }
        )
    return splits


def return_index(getitem):
    def with_index(self, index):
        res = getitem(self, index)
        return res + (index,)

    return with_index


@numba.jit(nopython=True, fastmath=True)
def hamming(x, y):
    """
    From https://github.com/talboger/fastdist/blob/master/fastdist/fastdist.py
    """
    n = len(x)
    num, denom = 0, 0
    for i in range(n):
        if x[i] != y[i]:
            num += 1
        denom += 1
    return num / denom


def get_pairwise_hausdorff_distances(concepts):
    dists = {}
    pairs = itertools.combinations_with_replacement(sorted(concepts.items()), 2)
    for (c1, a1), (c2, a2) in pairs:
        if (c1, c2) not in dists:
            dists[(c1, c2)] = hausdorff.hausdorff_distance(a1, a2, distance=hamming)
    return dists


def get_game_type(args):
    if args.reference_game:
        return "ref"
    elif args.percent_novel == 0.0:
        return "setref"
    elif args.percent_novel == 1.0:
        return "concept"
    else:
        return None
