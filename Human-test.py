import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from snare.datasets_zoo import data_des, get_dataset
from snare import set_seed, _default_collate, save_scores, datasets_zoo
from snare.models_zoo.vilt import collate
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/mnt/afs/wangfei154/project/SNARE/dataset", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="COCO_Semantic_Structure", type=str,
                        choices=["Attribute_Ownership", "Relationship_Composition",
                                 "Spatial_Relationship", "Negation_Logic",
                                 "COCO_Semantic_Structure", "Flickr30k_Semantic_Structure",])

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--text_perturb_fn", default=None, type=str,
                        help="Perturbation function to apply to the text.")
    parser.add_argument("--image_perturb_fn", default=None, type=str,
                        help="Perturbation function to apply to the images.")

    parser.add_argument("--download", action="store_true",
                        help="Download the datasets_zoo if it doesn't exist. (Default: False)")
    parser.add_argument("--save_scores", action="store_false",
                        help="Save the scores for the retrieval. (Default: True)")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    return parser.parse_args()


def deal_itm(itm, itype, num=None, shuffle=True):
    
    if shuffle:
        i = random.choice([0,1])
    else:
        i = 0 

    if itype == "Attribute_Ownership":
        # Each test case has a correct and incorrect caption.
        prompt = "### Does the following sentence match the image? 0 → no  1 → yes \n"
        if num in [0, 1]:
            prompt = "### Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
            true_caption = itm["true_caption"]
            false_caption = itm["false_caption"]
            sentence = [None, None]
            sentence[i] = f"{i}. {true_caption}"
            sentence[1-i] = f"{1-i}. {false_caption}"
            sentence = "\n".join(sentence)
        elif num in [2]:
            choice = [[itm["attributes"][1], itm["attributes"][0]], [itm["attributes"][0], itm["attributes"][1]]]
            form = [itm["obj1_name"], itm["obj2_name"]]
            form.extend(choice[i])
            sentence = "the {} and the {} are {} and {} respectively".format(*form)

    elif itype == "Negation_Logic":
        prompt = "### Does the following sentence match the image? 0 → no  1 → yes\n"
        if num == 0:
            choice = [[itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]], [itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]]]
            sentence = "the {} is {} and the {} is {}".format(*choice[i])
        elif num == 1:
            choice = [[itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]], [itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]]]
            sentence = "the {} is not {} and the {} is not {}".format(*choice[i])
        elif num == 2:
            prompt = "### Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
            choice = [[itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]], [itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]]]
            sentence_is = "the {} is {} and the {} is {}".format(*choice[i])
            sentence_not = "the {} is not {} and the {} is not {}".format(*choice[i])
            sentence = "0. " + sentence_is + "\n" + "1. " + sentence_not
    
    elif itype == "Relationship_Composition":
        prompt = "### Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"

        true_caption = itm["true_caption"]
        false_caption = itm["false_caption"]
        sentence = [true_caption, false_caption][i]
        blank_rela = itm["true_caption"].split(" is " + itm["relation_name"] + " ")
        blank_rela = " and ".join(blank_rela[::-1]) if i else " and ".join(blank_rela)
        sentence = "0. " + sentence + "\n" + "1. " + blank_rela

    elif itype == "Spatial_Relationship":
        prompt = "### Select the position information that best matches the image filled in the __. Respond using the corresponding option number.\n"
        sentence = itm["true_caption"].replace(itm["relation_name"], "__") + "\n" + "0. to the left of \n1. to the right of \n2. on \n3. below"
        i = ["to the left of", "to the right of", "on", "below"].index(itm["relation_name"])
        
    elif itype in ["Flickr30k_Semantic_Structure", "COCO_Semantic_Structure"]:
        prompt = "### Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
        i = random.choice(range(len(itm["caption_options"])))
        ll = itm["caption_options"][1:]
        ll.insert(i, itm["caption_options"][0])
        sentence = ""
        for j in range(len(ll)):
            sentence += f"{j}. {ll[j]} \n"

    return prompt, sentence, i # 0 is correct one, and 1 is wrong one.


def main(args):
    set_seed(args.seed)
    datasets_zoo.COCO_ROOT = os.path.join(args.data_path, "coco")
    datasets_zoo.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
    datasets_zoo.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")


    if args.dataset == "Flickr30k_Semantic_Structure":
        root_path = os.path.join(args.data_path, "flickr30k")
    elif args.dataset == "COCO_Semantic_Structure":
        root_path = os.path.join(args.data_path, "coco")
    

    # qustion = "are there {} in the image?"

    n = 100

    if args.dataset in ["COCO_Semantic_Structure", "Flickr30k_Semantic_Structure"]:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).test_cases
        
    else:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).dataset
    
    dataset = random.sample(dataset, n)

    cor = []
    hum = []
    sentences = []
    n_ = 1
    for itm in dataset:
        if args.dataset in ["COCO_Semantic_Structure", "Flickr30k_Semantic_Structure"]:
            image_path = os.path.join(root_path, itm["image"])
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(itm["image_path"]).convert('RGB')
            # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
            image = image.crop((itm["bbox_x"], itm["bbox_y"], itm["bbox_x"] + itm["bbox_w"],
                                itm["bbox_y"] + itm["bbox_h"]))

        image.show()
        # 创建一个图像窗口并显示图片
        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
        image.save("/mnt/afs/wangfei154/project/SNARE/img/linshi.png")
        prompt, sentence, i = deal_itm(itm, itype=args.dataset, num=args.num)
        print("*" * 10, f" {n_}/{n} ", "*" * 10)
        n_ += 1 
        print(prompt + sentence)

        output = input()
        while output not in ["0", "1", "2", "3"]:
            print("Please select from index.")
            output = input()
        
        hum.append(int(output))
        cor.append(i)
        sentences.append(sentence)

    save_dic = {"sentence": sentences, "ground_truth": cor, "human": hum}

    print(f"Acc: {np.count_nonzero(np.array(cor)==np.array(hum))/len(cor)}")

    output_file = os.path.join(args.output_dir, f"{args.dataset}_seed-{args.seed}_human_{args.num}.csv")

    df = pd.DataFrame(save_dic)
    os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)
    else:
        df.to_csv(output_file)

if __name__ == "__main__":
    args = config()
    main(args)
