import pdb

import numpy as np
import os
import torch
from PIL import Image
import pandas as pd
from collections import Counter


from . import util
from . import image_util as iu


IMAGE_SIZE = 224
LOAD_INTO_MEMORY = True


# FIXME - this is slightly incorrect since CUB classes are 1-indexed.
TRAIN_CLASSES = range(3)
VAL_CLASSES = range(3,5)
TEST_CLASSES = range(4, 6)

TRAIN_CLASSES_DEBUG = range(1)
VAL_CLASSES_DEBUG = range(1, 3)
TEST_CLASSES_DEBUG = range(3, 5)


def load_attr_dict(cub_path):
    cub_path='data/cub/'
    fpath = os.path.join(cub_path, "attributes.txt")
    attr_df = pd.read_csv(
        fpath,
        header=None,
        sep=" ",
        names=["attr_id", "attr_name"],
        usecols=["attr_name"],
    )
    # attr_id is 1-indexed but we want 0 index
    attrs = attr_df["attr_name"]

    attr_type_dict = {}
    attr_type_val_count_dict = Counter()
    attr_val_dict = {}

    for attr_i, attr in enumerate(attrs):
        attr_type, attr_val = attr.split("::")

        if attr_type not in attr_type_dict:
            attr_type_dict[attr_type] = len(attr_type_dict)

        attr_type_n = attr_type_dict[attr_type]

        attr_type_val_count = attr_type_val_count_dict[attr_type]

        # (which index, what value)
        attr_val_dict[attr_i] = (attr_type_n, attr_type_val_count)

        attr_type_val_count_dict[attr_type] += 1

    return attr_val_dict    # 根据CUB数据集的属性文件，创建一个属性字典。属性字典存储了属性索引与属性类型和属性值之间的映射关系。
    # 它还记录了每个属性类型的计数，以及为每个不同的属性类型分配的唯一索引。


def load_class_metadata(cub_path):
    md_path = os.path.join(
        cub_path, "attributes", "class_attribute_labels_continuous.txt"
    )
    md = pd.read_csv(md_path, sep=" ", header=None).to_numpy(dtype=np.float32)
    md = (md > 50.0).astype(np.uint8)
    # Now map to unique class names which are +1 indexed
    md_dict = {}
    for i in range(md.shape[0]):
        md_dict[i + 1] = md[i]
    # Now map from image ids to unique class names
    im_path = os.path.join(cub_path, "image_class_labels.txt")
    im2cl = pd.read_csv(im_path, sep=" ", header=None, names=["image_id", "class_id"])
    im2cl = dict(zip(im2cl["image_id"], im2cl["class_id"]))

    im_dict = {}
    for im_id, cl_id in im2cl.items():
        im_dict[im_id] = md_dict[cl_id]
    return im_dict   # 图像ID与类别属性标签的映射关系存储在字典中返回。


def load_img_metadata(cub_path):
    """
    TODO - do we do per class here?
    """
    md_path = os.path.join(cub_path, "attributes", "image_attribute_labels.txt")
    md = pd.read_csv(
        md_path,
        sep=" ",
        names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
        usecols=["is_present"],
        dtype={"is_present": np.uint8},
        header=None,
    )
    md_arr = md["is_present"].to_numpy()
    # Take slices by # of attributes, which is by 312
    md_dict = {}
    i = 1  # img id
    for start in range(0, len(md_arr), 312):
        md_dict[i] = md_arr[start : start + 312]
        i += 1

    # Now this is a mapping from each image id to the array
    return md_dict     # 字典的键是图像ID，值是一个数组，表示图像的属性存在性信息。


def load_cub_metadata(args):
    cub_dir = os.path.join(args.dataset, "CUB_200_2011")

    cub_dir='data/cub/CUB_200_2011/'
    # Load metadata per image
    img_md = load_img_metadata(cub_dir)
    class_md = load_class_metadata(cub_dir)

    # Load mapping from image names (npz keys) to metadata
    id2name = pd.read_csv(
        os.path.join(cub_dir, "images.txt"),
        sep=" ",
        names=["image_id", "name"],
        header=None,
    )
    id2name = dict(zip(id2name["image_id"], id2name["name"]))

    def rename_md(md):
        # Rename to image names
        md = {id2name[i]: m for i, m in md.items()}
        # Add path to cub dir so we can look up md in CUBDataset
        md = {
            os.path.join("cub", "CUB_200_2011", "images", k): v for k, v in md.items()
        }
        return md

    img_md = rename_md(img_md)   # 'cub\\CUB_200_2011\\images\\200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg': array()
    class_md = rename_md(class_md)

    return img_md, class_md


def load(args):
    if 'my_food' in args.dataset:
        img_dir=args.dataset
        print("Loading my food...")
    else:
        img_dir = os.path.join(args.dataset, "CUB_200_2011", "images")
        print("Loading CUB...")
    classes = os.listdir(img_dir)
    imgs = {}

    for cl in classes:    # 一个cl代表一个类别的名称
        cl_n = int(cl.split(".")[0])   # cl_n代表类别的编号
        if args.debug:
            if not any(
                cl_n in r
                for r in [TRAIN_CLASSES_DEBUG, VAL_CLASSES_DEBUG, TEST_CLASSES_DEBUG]
            ):                                       # 如果变量cl_n不是列表中任何一个集合的成员，则执行下一行代码
                continue
        npz_dir = os.path.join(img_dir, cl, "img.npz")
        if not os.path.exists(npz_dir):
            raise RuntimeError(
                f"Couldn't find {npz_dir}, run save_food_np.py in data/ first?"
            )
        cl_imgs = np.load(npz_dir)
        if LOAD_INTO_MEMORY:  # Load npz into memory
            cl_imgs = dict(cl_imgs)
        imgs[cl_n] = cl_imgs
    print("...done")

    # Load metadata
    img_md, class_md = load_cub_metadata(args)
    if args.reference_game:
        md = img_md     # 获取图像于属性映射关系
    else:
        md = class_md   # 获取类别于属性映射关系

    attr_dict = load_attr_dict(args.dataset)

    tloader = iu.TransformLoader(IMAGE_SIZE)
    train_transform = tloader.get_composed_transform(
        aug=True,
        normalize=True,
        to_pil=True,
    )
    # 数据增强
    test_transform = tloader.get_composed_transform(
        aug=True,
        normalize=True,
        to_pil=True,
    )   # 是否进行数据处理，增强、归一、转化为PIL

    def to_dset(classes, train=False, percent_novel=None, reference_game=None):
        if train:
            tr = train_transform
            length = 1000
        else:
            # tr = test_transform
            tr = train_transform
            length = 1000

        if args.debug:
            length = length // 10

        if percent_novel is None:
            percent_novel = args.percent_novel
        if reference_game is None:
            reference_game = args.reference_game

        # clases refers to the classes in loaded imgs
        subset = {k: v for k, v in imgs.items() if k in classes}
        return CUBDataset(
            subset,
            md,
            img_md,
            attr_dict,
            transform=tr,
            n_examples=args.n_examples,
            length=length,
            reference_game=reference_game,
            percent_novel=percent_novel,
        )

    if args.debug:
        classes = {
            "train": TRAIN_CLASSES_DEBUG,
            "val": VAL_CLASSES_DEBUG,
            "test": TEST_CLASSES_DEBUG,
        }
    else:   # classes: {'train': range(0, 3), 'val': range(3, 4), 'test': range(4, 5)}
        classes = {
            "train": TRAIN_CLASSES,
            "val": VAL_CLASSES,
            "test": TEST_CLASSES,
        }

    datasets = {
        "train": to_dset(classes["train"], train=True),
        "val": to_dset(classes["val"], train=False),
        "test": to_dset(classes["test"], train=False),
    }

    # Load other splits ?
    this_game_type = util.get_game_type(args)
    for _split in ["val", "test"]:
        for game_type in ["ref", "setref", "concept"]:
            split = f"{_split}_{game_type}"
            if game_type == this_game_type:
                datasets[split] = datasets[_split]
            else:
                datasets[split] = to_dset(
                    classes[_split],
                    percent_novel=1.0 if game_type == "concept" else 0.0,
                    reference_game=game_type == "ref",
                    train=False,
                )
    return datasets


class CUBDataset:
    MAX_N_EXAMPLES = 20
    name = "cub"
    meaning_distance_fn = "hamming"

    def __init__(
        self,
        imgs,
        metadata,
        img_metadata,
        attr_dict,
        n_examples=None,
        transform=None,
        length=1000,
        reference_game=False,
        percent_novel=1.0,
    ):
        self.imgs = imgs
        self.metadata = metadata
        self.img_metadata = img_metadata  # For hausdorff
        self.classes = np.array(list(self.imgs.keys()))
        self.img_names = {c: list(i.keys()) for c, i in self.imgs.items()}
        self.length = length
        self.transform = transform
        self.reference_game = reference_game
        self.n_feats = (3, IMAGE_SIZE, IMAGE_SIZE)
        self.attr_dict = attr_dict
        self.n_attr_types = max(t[0] for t in self.attr_dict.values()) + 1
        if n_examples is None:
            self.n_examples = 20
        else:
            self.n_examples = n_examples
        self.percent_novel = percent_novel

        # Make sure the metadata matches up
        # 由于windows和linux的区别，我先注释掉这里的代码！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # for c, ls in self.img_names.items():
        #     for l in ls:
        #         print(l)
        #         assert l in self.metadata.keys(),l
        #         assert l in self.metadata, l

        # for c, ls in self.img_names.items():
        #     for l in range(0, len(ls)):
        #         ls[l]=ls[l].replace("/", "\\", 4)

        # for c, inames in self.img_names.items():
        #     print(c),print(inames)
        # Get pairwise hausdorff distances for each class
        # self.concepts = {
        #     c: np.stack([self.img_metadata[n] for n in inames])
        #     for c, inames in self.img_names.items()
        # }
        # self.concept_distances = util.get_pairwise_hausdorff_distances(self.concepts)
        self.concept_distances = 0

    @util.return_index
    def __getitem__(self, i):
        """
        Get an item. Note the i doesn't matter, we just randomly sample.
        (Should this be the case for val? maybe not?)
        """
        return self.sample_game()

    def sample_negatives(self, n, pos_cl):
        neg_imgs = []
        neg_names=[]
        for _ in range(n):
            neg_cl = pos_cl
            while neg_cl == pos_cl:
                neg_cl = np.random.choice(self.classes)
            neg_img_name = np.random.choice(self.img_names[neg_cl])
            # Choose a cl
            neg_img = self.imgs[neg_cl][neg_img_name]
            # neg_img = self.imgs[neg_cl][neg_img_name.replace('\\', '/', 3)]
            neg_imgs.append(neg_img)
            neg_names.append(neg_img_name)
        return neg_imgs, neg_names

    def sample_game(self):
        # Randomly choose a class
        cl = np.random.choice(self.classes)
        if self.reference_game:
            # Select a single positive target
            pos_name = np.random.choice(self.img_names[cl])
            ex=1
            pos_imgs = [self.imgs[cl][pos_name] for _ in range(ex)]
            # md = self.metadata[pos_name]
            percent_novel = 0.0
        else:
            # pos_name = np.random.choice(
            #     self.img_names[cl], size=self.n_examples, replace=True
            # )
            pos_name = np.random.choice(
                self.img_names[cl], size=1, replace=True
            )
            pos_imgs = [self.imgs[cl][name] for name in pos_name]
            # pos_imgs = [self.imgs[cl][name.replace('\\', '/', 3)] for name in pos_names]    # 这里也替换了斜杠了
            # md = self.metadata[pos_names[0]]
            percent_novel = self.percent_novel

        neg_imgs, neg_names = self.sample_negatives(self.n_examples-1, cl)
        names =[pos_name, neg_names]

        tr_ratio=[]
        for img in pos_imgs:
            ratio= [224 / img.shape[0], 224 / img.shape[1]]
            ratio=tuple(ratio)
            tr_ratio.append(ratio)
        for img in neg_imgs:
            ratio= [224 / img.shape[0], 224 / img.shape[1]]
            ratio=tuple(ratio)
            tr_ratio.append(ratio)

        if self.transform is not None:
            pos_imgs = [self.transform(img) for img in pos_imgs]
            neg_imgs = [self.transform(img) for img in neg_imgs]
        else:
            # Convert to tensor
            raise NotImplementedError

        imgs, y = util.stack_pos_neg(pos_imgs, neg_imgs)   # imgs 维度: 12*3*224*224

        # 0th metadata is a game indicator, which we don't use
        # md = torch.from_numpy(md)
        # diff = torch.zeros((1,), dtype=md.dtype)
        # md = torch.cat([diff, md], 0)
        # "padding"
        txt = np.full(3, cl, dtype=np.int64)
        splits = util.split_spk_lis(
            imgs, y, self.n_examples, names, tr_ratio, percent_novel=percent_novel
        )
        return splits + (txt, self.metadata)

    def __len__(self):
        return self.length

    def vis_input(self, inp, overwrite=True, **kwargs):
        img_fname = f"{kwargs['name']}_{kwargs['epoch']}_{kwargs['split']}_{kwargs['game_i']}_{kwargs['i']}.jpg"
        img_f = os.path.join(kwargs["exp_dir"], "images", img_fname)
        img_html = f"""<img src="{os.path.join('images', img_fname)}">"""
        if os.path.exists(img_f) and not overwrite:
            return img_html

        inp = iu.unnormalize_t_(inp).permute((1, 2, 0)).numpy()
        inp = np.round(inp * 224).astype(np.uint8)
        Image.fromarray(inp).save(img_f)
        return img_html

    def to_text(self, idxs, join=True):
        texts = []
        for lang in idxs:
            toks = []
            toks.append("<s>")
            for i in lang[1:-1]:
                toks.append(str(i.item()))
            toks.append("</s>")
            if join:
                texts.append(" ".join(toks))
            else:
                texts.append(toks)
        return texts

    def attr_to_numeric(self, attrs):
        """
        Convert the length-312 one-hot attributes to standard numeric
        attributes
        """
        batch_size = len(attrs)
        n_attrs = attrs[0].shape[0]
        assert n_attrs == len(self.attr_dict)
        attrs_numeric = []

        for i in range(batch_size):
            attr_num_i = np.zeros(self.n_attr_types, dtype=np.int64)
            for j in range(n_attrs):
                if attrs[i][j]:
                    attr_type_i, attr_val = self.attr_dict[j]
                    attr_num_i[attr_type_i] = attr_val
            attrs_numeric.append(attr_num_i)

        return attrs_numeric

    def concept_distance(self, c1, c2):
        pair = tuple(sorted((c1, c2)))
        return self.concept_distances[pair]
