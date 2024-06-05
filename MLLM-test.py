import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from snare.datasets_zoo import data_des, get_dataset
from snare import set_seed, _default_collate, save_scores, datasets_zoo
from snare.models_zoo.vilt import collate
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import random

from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/mnt/afs/wangfei154/project/SNARE/dataset", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="Attribute_Ownership", 
                    type=str,
                    choices=["Attribute_Ownership", "Relationship_Composition", 
                            "Spatial_Relationship", "Negation_Logic",
                            "COCO_Semantic_Structure", "Flickr30k_Semantic_Structure",
                            "VG_Relation", "VG_Attribution", "COCO:_Order", "Flickr30k_Order"])
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--download", action="store_true",
                        help="Download the datasets_zoo if it doesn't exist. (Default: False)")
    parser.add_argument("--save_scores", action="store_false",
                        help="Save the scores for the retrieval. (Default: True)")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    return parser.parse_args()


import base64
from mimetypes import guess_type


def get_mllm_resut(data, model, processors, device="cuda:0"):
    inputs = processors(images=data["image"], text=data["text"], return_tensors="pt").to(device=device, dtype=torch.bfloat16)
    description = model.generate(**inputs, max_length=100)
    description = processors.batch_decode(description, skip_special_tokens=True)[0].strip()
    return description

def deal_itm_human(itm, itype, num=None, shuffle=True):
    
    if shuffle:
        i = random.choice([0,1])
    else:
        i = 0 

    if itype == "Attribute_Ownership":
        # Each test case has a correct and incorrect caption.
        prompt = "Does the following sentence match the image? 0 → no  1 → yes \n"
        if num in [0]:
            prompt = "Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
            true_caption = itm["true_caption"]
            false_caption = itm["false_caption"]
            sentence = [None, None]
            sentence[i] = f"{i}. {true_caption}"
            sentence[1-i] = f"{1-i}. {false_caption}"
            sentence = "\n".join(sentence)
        elif num in [1]:
            prompt = "Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
            caption = [itm["true_caption"], itm["false_caption"]]
            sentence = [None, None]
            sentence[0] = f"{0}. {caption[i]}"
            choice = [[itm["attributes"][1], itm["attributes"][0]], [itm["attributes"][0], itm["attributes"][1]]]
            form = [itm["obj1_name"], itm["obj2_name"]]
            form.extend(choice[i])
            sentence[1] = "1. the {} and the {} are {} and {} respectively".format(*form)
            sentence = "\n".join(sentence)
        elif num in [2]:
            choice = [[itm["attributes"][1], itm["attributes"][0]], [itm["attributes"][0], itm["attributes"][1]]]
            form = [itm["obj1_name"], itm["obj2_name"]]
            form.extend(choice[i])
            sentence = "the {} and the {} are {} and {} respectively".format(*form)
        else:
            assert 0

    elif itype == "Negation_Logic":
        prompt = "Does the following sentence match the image? Answer yes or no.\n"
        if num == 0:
            choice = [[itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]], [itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]]]
            sentence = "the {} is {} and the {} is {}".format(*choice[i])
        elif num == 1:
            choice = [[itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]], [itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]]]
            sentence = "the {} is not {} and the {} is not {}".format(*choice[i])
        elif num == 2:
            prompt = "Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
            choice = [[itm["obj1_name"], itm["attributes"][0], itm["obj2_name"], itm["attributes"][1]], [itm["obj1_name"], itm["attributes"][1], itm["obj2_name"], itm["attributes"][0]]]
            sentence_is = "the {} is {} and the {} is {}".format(*choice[i])
            sentence_not = "the {} is not {} and the {} is not {}".format(*choice[i])
            sentence = "0. " + sentence_is + "\n" + "1. " + sentence_not
        else:
            assert 0
    
    elif itype == "Relationship_Composition":
        assert num == 0
        prompt = "Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"

        true_caption = itm["true_caption"]
        false_caption = itm["false_caption"]
        sentence = [true_caption, false_caption][i]
        blank_rela = itm["true_caption"].replace("is " + itm["relation_name"], "and")
        sentence = "0. " + sentence + "\n" + "1. " + blank_rela

    elif itype == "Spatial_Relationship":
        assert num == 0
        prompt = "Select the position information that best matches the image filled in the __. Respond using the corresponding option number.\n"
        sentence = itm["true_caption"].replace(itm["relation_name"], "__") + "\n" + "0. to the left of \n1. to the right of \n2. on \n3. below"
        i = ["to the left of", "to the right of", "on", "below"].index(itm["relation_name"])
        
    elif itype in ["Flickr30k_Semantic_Structure", "COCO_Semantic_Structure"]:
        assert num == 0
        prompt = "Select the most accurate description for the image from the following options. Respond using the corresponding option number.\n"
        i = random.choice(range(len(itm["caption_options"])))
        ll = itm["caption_options"][1:]
        ll.insert(i, itm["caption_options"][0])
        sentence = ""
        for j in range(len(ll)):
            sentence += f"{j}. {ll[j]} \n"

    return prompt, sentence, i # 0 is correct one, and 1 is wrong one.

def get_model(device="cuda:0", root_dir="/mnt/afs/wangfei154/project/SNARE/models"):
    processors = AutoProcessor.from_pretrained(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl") if os.path.exists(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl")) else "Salesforce/blip2-flan-t5-xl",)
    model = AutoModelForVisualQuestionAnswering.from_pretrained(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl") if os.path.exists(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl")) else "Salesforce/blip2-flan-t5-xl", device_map=device, torch_dtype=torch.float16, cache_dir=root_dir)
    model.eval()
    return model, processors

def main(args):
    set_seed(args.seed)
    datasets_zoo.COCO_ROOT = os.path.join(args.data_path, "coco")
    datasets_zoo.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
    datasets_zoo.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")


    if args.dataset == "Flickr30k_Semantic_Structure":
        root_path = os.path.join(args.data_path, "flickr30k")
    elif args.dataset == "COCO_Semantic_Structure":
        root_path = os.path.join(args.data_path, "coco")

    if args.dataset in ["COCO_Semantic_Structure", "Flickr30k_Semantic_Structure"]:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).test_cases
    else:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).dataset
    
    n = len(dataset)
    len_samples = len(dataset)
    dataset = random.sample(dataset, n)
    print(f"sample {len(dataset)} from dataset ({len_samples})")

    model, processors = get_model()

    cor = []
    hum = []
    sentences = []
    responce = []
    nn = 0
    tt = 0
    for itm in dataset:
        data = {}
        if args.dataset in ["COCO_Semantic_Structure", "Flickr30k_Semantic_Structure"]:
            image_path = os.path.join(root_path, itm["image"])
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                print("Can't find image.")
                n -= 1
                continue
        else:
            image = Image.open(itm["image_path"]).convert('RGB')
            # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
            image = image.crop((itm["bbox_x"], itm["bbox_y"], itm["bbox_x"] + itm["bbox_w"],
                                itm["bbox_y"] + itm["bbox_h"]))


        prompt, sentence, i = deal_itm_human(itm, itype=args.dataset, num=args.num)
        data["text"] = prompt + sentence
        data["image"] = image

        print("*" * 20)
        print(f"{nn+1}/{n}, ACC:{tt}/{nn}", prompt + sentence)
        nn += 1

        output_ = get_mllm_resut(data, model, processors)
        if output_ != None:
            output_ = output_.lower()
            output = None
            if "yes" in output_:
                output = 1
            elif "no" in output_:
                output = 0
            else:
                for j in ["0", "1", "2", "3", "4"]:
                    if j in output_:
                        output = int(j)
            if output == None:
                print(output_)
                output = 100

            print(output)
            
            hum.append(int(output))
            cor.append(i)
            sentences.append(sentence)
            responce.append(output_)
            if i == output:
                tt += 1
            
        else:
            nn -= 1
            n -= 1
            print("无输出")
            continue

    save_dic = {"sentence": sentences, "responce": responce, "ground_truth": cor, "mllm": hum}

    acc = np.count_nonzero(np.array(cor)==np.array(hum))/len(cor)
    print(f"Acc: {acc}")
    save_dic["mllm"].append(acc)
    save_dic["sentence"].append("Acc")
    save_dic["responce"].append(n)
    save_dic["ground_truth"].append(tt)

    output_file = os.path.join(args.output_dir, f"{args.dataset}_seed-{args.seed}_blip2_{args.num}.csv")

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
