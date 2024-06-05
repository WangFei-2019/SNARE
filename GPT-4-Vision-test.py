import argparse
import os
import pandas as pd

from sympy import print_rcode, root
from torch.utils.data import DataLoader

from snare.datasets_zoo import data_des, get_dataset
from snare import set_seed, _default_collate, save_scores, datasets_zoo
from snare.models.vilt import collate
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random

import json
import requests
import openai
import logging
import time
import concurrent.futures

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/workspace/public/data0/HOME/jdnlp1004/wangfei154/project/SNARE/dataset", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="Attribute_Ownership", 
                    type=str,
                    choices=["Attribute_Ownership", "Relationship_Composition", 
                            "Spatial_Relationship", "Negation_Logic",
                            "COCO_Semantic_Structure", "Flickr30k_Semantic_Structure",
                            "VG_Relation", "VG_Attribution", "COCO:_Order", "Flickr30k_Order"])
    parser.add_argument("--num", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--download", action="store_true",
                        help="Download the datasets_zoo if it doesn't exist. (Default: False)")
    parser.add_argument("--save_scores", action="store_false",
                        help="Save the scores for the retrieval. (Default: True)")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    return parser.parse_args()


import base64
from mimetypes import guess_type

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def get_gpt4_respond(account_list, data):
    system_message = "You are a helpful assistant."
    # 设置OpenAI API credentials
    openai.api_key = account_list["api_key"] # os.getenv("OPENAI_API_KEY")
    openai.api_base = account_list["api_base"] # os.getenv("OPENAI_API_BASE")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }


    # 本示例为请求聊天完成接口，如果需要请求别的接口请修改
    print(80*'-')
    gpt_message = [{"role": "system", "content": system_message}, {"role": "user", "content": [{"type":"text", "text": data["text"]}, {"type": "image_url", "image_url": {"url": data["image"]}}]}]
    #print('messages:', messages[0])

    test = 0
    sleep_time = 2
    while test < 5:
        try:
            response = openai.ChatCompletion.create(
                model= "gpt-4-vision-preview", # "gpt-4-1106-preview", # "gpt-4-vision-preview",# "gpt-35-turbo-1106", # "gpt-4-1106-preview",
                messages=gpt_message,
                temperature=0.5,
                max_tokens=128,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False,
                # 入参时erp改为不必填 但如果输入了erp会校验erp是否真实存在，输入erp与计费相关（不输入则使用申请人erp结算），如果输入erp不对调用会报错哦
                # erp=" ",
                headers=headers
            )

            # print(80*'*')
            # print('responese: ', response)
            output = response['choices'][0]['message']['content']
            return output
        except:
            time.sleep(sleep_time)
            test += 1
            return None

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
    account_list = {
            "api_base": "Yours",
            "api_key": "Yours"
        }

    n = 1500
    num = args.num

    if args.dataset in ["COCO_Semantic_Structure", "Flickr30k_Semantic_Structure"]:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).test_cases
    else:
        dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download).dataset
    
    len_samples = len(dataset)
    dataset = random.sample(dataset, n)
    print(f"sample {len(dataset)} from dataset ({len_samples})")

    cor = []
    hum = []
    sentences = []
    responce = []
    image_path = f"/workspace/public/data0/HOME/jdnlp1004/wangfei154/project/SNARE/{args.dataset}{args.num}.png"
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
            if os.path.exists(image_path):
                os.remove(image_path)
            image = Image.open(itm["image_path"]).convert('RGB')
            # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
            image = image.crop((itm["bbox_x"], itm["bbox_y"], itm["bbox_x"] + itm["bbox_w"],
                                itm["bbox_y"] + itm["bbox_h"]))
            image.save(image_path)
        image_url = local_image_to_data_url(image_path)


        prompt, sentence, i = deal_itm_human(itm, itype=args.dataset, num=num)
        data["text"] = prompt + sentence
        data["image"] = image_url

        print("*" * 20)
        print(f"{nn+1}/{n}, ACC:{tt}/{nn}", prompt + sentence)
        nn += 1

        output_ = get_gpt4_respond(account_list, data)
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
            print("请求失败")
            continue

    save_dic = {"sentence": sentences, "responce": responce, "ground_truth": cor, "human": hum}

    acc = np.count_nonzero(np.array(cor)==np.array(hum))/len(cor)
    print(f"Acc: {acc}")
    save_dic["human"].append(acc)
    save_dic["sentence"].append("Acc")
    save_dic["responce"].append(n)
    save_dic["ground_truth"].append(nn)

    output_file = os.path.join(args.output_dir, f"{args.dataset}_seed-{args.seed}_gpt4_{args.num}.csv")

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
    # image_path = '<path_to_image>'
    # data_url = local_image_to_data_url(image_path)
    # print("Data URL:", data_url)
