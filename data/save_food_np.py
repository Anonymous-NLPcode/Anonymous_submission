"""
For each class, load images and save as numpy arrays.
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from kgfunction import KGtovec

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Save numpy", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cub_dir", default="my_food", help="Directory to load/cache"
    )
    args = parser.parse_args()


    # from lavis.models import load_model_and_preprocess
    # device_vqa = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_vqa, vis_processors_vqa, txt_processors_vqa = load_model_and_preprocess(name="blip_vqa", model_type="vqav2",
    #                                                                               is_eval=True,
    #                                                                               device=device_vqa)
    # from gensim.models import KeyedVectors
    # model_w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    for bird_class in tqdm(os.listdir(args.cub_dir), desc="Classes"):
        bird_imgs_np = {}
        graph_dict = {}
        class_dir = os.path.join(args.cub_dir, bird_class)
        bird_imgs = sorted([x for x in os.listdir(class_dir) if x != "img.npz"])
        # file_path = "graph_dict"+bird_class+'.csv'
        for bird_img in bird_imgs:
            bird_img_fname = os.path.join(class_dir, bird_img)
            # graph=KGtovec.graph2vec2(bird_img_fname,model_vqa,vis_processors_vqa,model_w2v)
            # graph_dict[bird_img_fname]=graph
            # torch.set_printoptions(threshold=len(graph))
            # with open(file_path, "w") as file:
            #     for key, value in graph_dict.items():
            #         file.write(f"{key}: {value}\n")
            img = Image.open(bird_img_fname).convert("RGB")
            img_np = np.asarray(img)

            full_bird_img_fname = os.path.join(
                args.cub_dir, bird_class, bird_img
            )
            bird_imgs_np[full_bird_img_fname] = img_np

        np_fname = os.path.join(class_dir, "img.npz")
        np.savez_compressed(np_fname, **bird_imgs_np)

    # 指定文件夹路径
    folder_path = "my_food"

    # 获取文件夹下所有文件名称
    file_names = os.listdir(folder_path)
