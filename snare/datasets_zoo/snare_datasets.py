import os
import json
import subprocess

import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url

from .data_des import Text_Des
from . import CASSP_ROOT, COCO_ROOT, FLICKR_ROOT
from .retrieval_dataset import pre_caption
from collections import Counter
from .utils import top_n_accuracy


class VG_Relation(Dataset):
	def __init__(self, image_preprocess, subordination_relation=False, multi_spatial_relation=False,
				 root_dir=CASSP_ROOT,
				 download=False, *args, **kwargs):
		'''
		image_preprocess: a function that takes in a PIL image and returns a tensor.
		text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
		image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
		root_dir: Directory for the VG-R dataset.
		download: Whether to download the dataset if it does not exist.
		'''
		self.root_dir = root_dir
		self.subordination_relation = subordination_relation
		self.multi_spatial_relation = multi_spatial_relation
		assert self.subordination_relation * self.multi_spatial_relation == 0
		annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
		image_dir = os.path.join(root_dir, "images")
		if not os.path.exists(image_dir):
			print("Image Directory for VG_Relation could not be found!")
			if download:
				self.download()
			else:
				raise RuntimeError(
					"Please either download the dataset by letting `--download` or specify the correct directory.")

		if not os.path.exists(annotation_file):
			subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])

		with open(annotation_file, "r") as f:
			self.dataset = json.load(f)

		self.top_2 = False
		self.all_relations = list()
		self.dataset_ = list()
		self.all_relations_ = list()
		self.targets_mul = list()
		relations = ["to the left of", "to the right of", "on", "below"] \
			if self.multi_spatial_relation else ["near", "next to", "beside", "on the side of", "surrounding",
												 "close to", "standing beside", "with", "by"]
		for item in self.dataset:
			item["image_path"] = os.path.join(image_dir, item["image_path"])
			if (item["relation_name"] in relations) and self.multi_spatial_relation:
				self.dataset_.append(item)
				self.all_relations.append(item["relation_name"])
			elif not self.multi_spatial_relation and (item["relation_name"] not in relations):
				self.dataset_.append(item)
				self.all_relations.append(item["relation_name"])
		self.dataset = self.dataset_
		if self.multi_spatial_relation:
			self.cla_name = relations
			self.top_2 = True
			for i in self.all_relations:
				label = np.zeros(len(relations))
				label[relations.index(i)] = 1
				self.targets_mul.append(label)
			self.targets_mul = np.array(self.targets_mul)
		elif self.subordination_relation:
			self.cla_name = ["correct", "exchange", "and"]
			self.cla_name = ["ARO", "(,,)", ",,"]
			self.top_2 = True
		else:
			self.cla_name = ["correct", "exchange"]
		self.targets = [np.repeat(np.diag(np.ones(len(self.cla_name)))[i][None, :], len(self.dataset), axis=0)
						for i in range(len(self.cla_name))]
		self.image_preprocess = image_preprocess

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		test_case = self.dataset[index]
		image = Image.open(test_case["image_path"]).convert('RGB')
		# Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
		image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"],
							test_case["bbox_y"] + test_case["bbox_h"]))

		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		if self.subordination_relation:
			# Each test case has a correct and incorrect caption.
			true_caption = test_case["true_caption"]
			false_caption = test_case["false_caption"]
			blank_rela = test_case["true_caption"].replace("is " + test_case["relation_name"], "and")
			caption_options = [true_caption, false_caption, blank_rela]

		elif self.multi_spatial_relation:
			caption_options = [test_case["true_caption"].replace(test_case["relation_name"], i) for i in self.cla_name]
		else:
			true_caption = test_case["true_caption"]
			false_caption = test_case["false_caption"]
			caption_options = [true_caption, false_caption]
		item = dict({"image_options": [image], "caption_options": caption_options})
		return item

	def download(self):
		os.makedirs(self.root_dir, exist_ok=True)
		image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
		subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
		subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

	def evaluate_scores(self, scores):
		"""
		Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
		"""
		if isinstance(scores, tuple):
			scores_i2t = scores[1]
			scores_t2i = scores[0]
		else:
			scores_t2i = scores
			scores_i2t = scores

		score = np.squeeze(scores_i2t, axis=1)
		top_1_dict = {self.cla_name[i]: top_n_accuracy(score, self.targets[i], 1) for i in
					  range(len(self.cla_name))}
		if self.top_2:
			top_2_list = {self.cla_name[i]: top_n_accuracy(score, self.targets[i], 2) for i in
						  range(len(self.cla_name))}

		result_records = []
		# 子类别数据
		all_relations = np.array(self.all_relations)
		for rela in np.unique(all_relations):
			rela_mask = (all_relations == rela)
			score_sub = score[rela_mask]
			if rela_mask.sum() < 25:
				continue
			res_dict = {
				"Attributes": rela,
				"Count": rela_mask.sum(),
			}
			for i in range(len(self.cla_name)):
				res_dict.update(
					{self.cla_name[i] + "_top-1": top_n_accuracy(score_sub, self.targets[i][rela_mask], 1)[0]})
				if self.top_2:
					res_dict.update(
						{self.cla_name[i] + "_top-2": top_n_accuracy(score_sub, self.targets[i][rela_mask], 2)[0]})
			result_records.append(res_dict)

		# 总体数据
		if self.multi_spatial_relation:
			top_1 = top_n_accuracy(score, self.targets_mul, 1)
			res_dict = {
				"Attributes": "clasfication",
				"top-1": top_1[0],
				"top-2": top_n_accuracy(score, self.targets_mul, 2)[0],
				"top-3": top_n_accuracy(score, self.targets_mul, 3)[0],
				"Count": top_1[1],
			}
			result_records.append(res_dict)
		else:
			for key, value in top_1_dict.items():
				res_dict = {
					"Attributes": key,
					key + "_top-1": value[0],
					"Count": value[1],
				}
				print(key, value[0])
				if self.top_2:
					res_dict.update({key + "_top-2": top_2_list[key][0]})
				result_records.append(res_dict)
		return result_records

	def evaluate_vllm_scores(self, scores):
		"""
		Scores: M x 1 x N, i.e. first caption is the perturbed one, second is the positive one
		"""
		if isinstance(scores, tuple):
			scores_i2t = scores[1]
			scores_t2i = scores[0]
		else:
			scores_t2i = scores
			scores_i2t = scores

		score = scores_i2t
		cla_acc = {self.cla_name[i]: sum(score[:, i])/len(score) for i in range(len(self.cla_name))}
		result_records = []
		# 子类别数据
		all_relations = np.array(self.all_relations)
		for attr in np.unique(all_relations):
			attr_mask = (all_relations == attr)
			score_sub = score[attr_mask]
			if attr_mask.sum() < 25:
				continue
			res_dict = {
				"Attributes": attr,
				"Count": attr_mask.sum(),
			}
			for i in range(len(self.cla_name)):
				res_dict.update({self.cla_name[i]: sum(score_sub[:, i])/len(score_sub)})
			result_records.append(res_dict)

		# 总体数据
		for key, value in cla_acc.items():
			res_dict = {
				"Attributes": key,
				key: value,
			}
			result_records.append(res_dict)
		return result_records


class VG_Attribution(Dataset):
	def __init__(self, image_preprocess, logic=False, attribute_ownership=False, root_dir=CASSP_ROOT, download=False,
				 *args, **kwargs):
		'''
		image_preprocess: a function that takes in a PIL image and returns a tensor.
		text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
		image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
		root_dir: Directory for the VG-A dataset.
		'''
		self.root_dir = root_dir
		self.logic = logic
		self.attribute_ownership = attribute_ownership
		annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
		image_dir = os.path.join(root_dir, "images")
		if not os.path.exists(image_dir):
			print("Image Directory for VG_Attribution could not be found!")
			if download:
				self.download()
			else:
				raise RuntimeError(
					"Please either download the dataset by letting `--download` or specify the correct directory.")

		if not os.path.exists(annotation_file):
			subprocess.call(["gdown", "--id", "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", "--output", annotation_file])

		with open(annotation_file, "r") as f:
			self.dataset = json.load(f)

		for item in self.dataset:
			item["image_path"] = os.path.join(image_dir, item["image_path"])

		# Set of attributes in each test case
		self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
		self.image_preprocess = image_preprocess
		self.top_2 = False
		if self.attribute_ownership:
			self.cla_name = ["exchange", "separate"]
			self.top_2 = True
		elif self.logic:
			self.cla_name = ["negative"]
		else:
			self.cla_name = ["exchange"]

		self.cla_name.insert(0, "correct")
		self.targets = [np.repeat(np.diag(np.ones(len(self.cla_name)))[i][None, :], len(self.dataset), axis=0)
						for i in range(len(self.cla_name))]

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		test_case = self.dataset[index]
		image = Image.open(test_case["image_path"]).convert('RGB')
		# Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
		image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"],
							test_case["bbox_y"] + test_case["bbox_h"]))

		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		if self.attribute_ownership:
			# Each test case has a correct and incorrect caption.
			true_caption = test_case["true_caption"]
			false_caption = test_case["false_caption"]
			split_semantic = "the {} and the {} are {} and {} respectively".format(test_case["obj1_name"],
																				   test_case["obj2_name"],
																				   test_case["attributes"][0],
																				   test_case["attributes"][1])
			caption_options = [true_caption, false_caption, split_semantic]
		elif self.logic:
			positive = "the {} is {} and the {} is {}".format(test_case["obj1_name"], test_case["attributes"][0],
															  test_case["obj2_name"], test_case["attributes"][1])
			negative = "the {} is not {} and the {} is not {}".format(test_case["obj1_name"],
																	  test_case["attributes"][0],
																	  test_case["obj2_name"],
																	  test_case["attributes"][1])
			caption_options = [positive, negative]
		else:
			true_caption = test_case["true_caption"]
			false_caption = test_case["false_caption"]
			caption_options = [true_caption, false_caption]
		item = edict({"image_options": [image], "caption_options": caption_options})
		return item

	def download(self):
		os.makedirs(self.root_dir, exist_ok=True)
		image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
		subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
		subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

	def evaluate_scores(self, scores):
		"""
		Scores: M x 1 x N, i.e. first caption is the perturbed one, second is the positive one
		"""
		if isinstance(scores, tuple):
			scores_i2t = scores[1]
			scores_t2i = scores[0]
		else:
			scores_t2i = scores
			scores_i2t = scores

		score = np.squeeze(scores_i2t, axis=1)
		top_1_dict = {self.cla_name[i]: top_n_accuracy(score, self.targets[i], 1) for i in
					  range(len(self.cla_name))}
		if self.top_2:
			top_2_list = {self.cla_name[i]: top_n_accuracy(score, self.targets[i], 2) for i in
						  range(len(self.cla_name))}

		result_records = []
		# 子类别数据
		all_attributes = np.array(self.all_attributes)
		for attr in np.unique(all_attributes):
			attr_mask = (all_attributes == attr)
			score_sub = score[attr_mask]
			if attr_mask.sum() < 25:
				continue
			res_dict = {
				"Attributes": attr,
				"Count": attr_mask.sum(),
			}
			for i in range(len(self.cla_name)):
				res_dict.update({self.cla_name[i] + "_top-1": top_n_accuracy(score_sub, self.targets[i][attr_mask], 1)[0]})
				if self.top_2:
					res_dict.update({self.cla_name[i] + "_top-2": top_n_accuracy(score_sub, self.targets[i][attr_mask], 2)[0]})
			result_records.append(res_dict)

		# 总体数据
		for key, value in top_1_dict.items():
			res_dict = {
				"Attributes": key,
				key+"_top-1": value[0],
				"Count": value[1],
			}
			print(key, value[0])
			if self.top_2:
				res_dict.update({key+"_top-2": top_2_list[key][0]})
			result_records.append(res_dict)
		return result_records

	def evaluate_vllm_scores(self, scores):
		"""
		Scores: M x 1 x N, i.e. first caption is the perturbed one, second is the positive one
		"""
		if isinstance(scores, tuple):
			scores_i2t = scores[1]
			scores_t2i = scores[0]
		else:
			scores_t2i = scores
			scores_i2t = scores

		score = scores_i2t
		cla_acc = {self.cla_name[i]: sum(score[:, i])/len(score) for i in range(len(self.cla_name))}
		result_records = []
		# 子类别数据
		all_attributes = np.array(self.all_attributes)
		for attr in np.unique(all_attributes):
			attr_mask = (all_attributes == attr)
			score_sub = score[attr_mask]
			if attr_mask.sum() < 25:
				continue
			res_dict = {
				"Attributes": attr,
				"Count": attr_mask.sum(),
			}
			for i in range(len(self.cla_name)):
				res_dict.update({self.cla_name[i]: sum(score_sub[:, i])/len(score_sub)})
			result_records.append(res_dict)

		# 总体数据
		for key, value in cla_acc.items():
			res_dict = {
				"Attributes": key,
				key: value,
			}
			result_records.append(res_dict)
		return result_records


class COCO_Semantic_Structure(Dataset):
	def __init__(self, image_preprocess=None, semantic_structure=True, root_dir=COCO_ROOT, max_words=30, split="test",
				 download=False):
		"""
		COCO Order Dataset.
		image_preprocess: image preprocessing function
		root_dir: The directory of the coco dataset. This directory should contain test2014 files.
		max_words: Cropping the caption to max_words.
		split: 'val' or 'test'
		image_perturb_fn: not used; for compatibility.
		download: Whether to download the dataset if it does not exist.
		"""
		test_des = Text_Des()
		self.semantic_structure = semantic_structure
		if self.semantic_structure:
			# perturb_functions = [test_des.shuffle_nouns_and_verb_adj, test_des.shuffle_allbut_nouns_verb_adj,
			# 					 test_des.shuffle_all_words, test_des.shuffle_nouns, test_des.shuffle_verb,
			# 					 test_des.shuffle_adj]
			perturb_functions = [test_des.shuffle_nouns_and_verb_adj, test_des.shuffle_allbut_nouns_verb_adj,
								 test_des.shuffle_all_words]
		else:
			perturb_functions = [test_des.shuffle_nouns_and_adj, test_des.shuffle_allbut_nouns_and_adj,
								 test_des.shuffle_within_trigrams, test_des.shuffle_trigrams, ]

		self.cla_name = [i.__name__ for i in perturb_functions]
		self.root_dir = root_dir
		if not os.path.exists(root_dir):
			print("Directory for COCO could not be found!")
			if download:
				print("Downloading COCO now.")
				self.download()
			else:
				raise RuntimeError(
					"Please either download the dataset by letting `--download` or specify the correct directory.")

		urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
				'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
		filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}
		download_url(urls[split], root_dir)

		self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
		self.image_preprocess = image_preprocess
		self.image_root = root_dir

		self.test_cases = []

		for img_id, ann in tqdm(enumerate(self.annotation), total=len(self.annotation), desc="Text Destroy Progress"):
			for i, caption in enumerate(ann['caption']):
				test_case = {}
				test_case["image"] = ann["image"]
				test_case["caption_options"] = []  # [pre_caption(caption, max_words)]

				for perturb_fn in perturb_functions:
					test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
				self.test_cases.append(test_case)

	def __len__(self):
		return len(self.test_cases)

	def __getitem__(self, index):
		test_case = self.test_cases[index]
		image_path = os.path.join(self.image_root, test_case["image"])

		image = Image.open(image_path).convert('RGB')
		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
		return item

	def download(self):
		import subprocess
		os.makedirs(self.root_dir, exist_ok=True)
		# subprocess.call(["wget", "http://images.cocodataset.org/zips/train2014.zip"], cwd=self.root_dir)
		# subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)

		subprocess.call(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=self.root_dir)
		subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)

		subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
		subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)

	def evaluate_scores(self, scores):
		if isinstance(scores, tuple):
			scores_i2t = scores[0]
			scores_t2i = scores[1].T  # Make it N_ims x N_text
		else:
			scores_t2i = scores
			scores_i2t = scores

		preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
		count = dict(Counter(preds))
		# correct_mask = (preds == 0)
		result_records = []

		for ids, pr in count.items():
			if ids != 0:
				rela = self.cla_name[ids - 1]
			else:
				rela = "Correct"
			result_records.append({
				"Relation": rela,
				"Accuracy": pr / len(preds),
				"Count": pr,
				"Dataset": "Visual Genome Relation"
			})
			print(rela,  pr / len(preds))
		return result_records


class Flickr30k_Semantic_Structure(Dataset):
	def __init__(self, image_preprocess, split, root_dir=FLICKR_ROOT, max_words=30, semantic_structure=True,
				 *args, **kwargs):
		"""
		image_preprocess: image preprocessing function
		split: 'val' or 'test'
		root_dir: The directory of the flickr30k images. This should contain the `flickr30k-images` directory that \
			contains all the images.
		"""
		urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
				'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
		filenames = {'val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}
		if not os.path.exists(root_dir):
			print("Directory for Flickr30k could not be found!")
			flickr_url = "https://forms.illinois.edu/sec/229675"
			raise RuntimeError(
				f"You need to manually sign up and download the dataset from {flickr_url} and place it in the `root_dir`.")

		download_url(urls[split], root_dir)

		self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
		self.image_preprocess = image_preprocess
		self.root_dir = root_dir
		self.semantic_structure = semantic_structure

		self.test_cases = []

		test_des = Text_Des()
		if self.semantic_structure:
			# perturb_functions = [test_des.shuffle_nouns_and_verb_adj, test_des.shuffle_allbut_nouns_verb_adj,
			# 					 test_des.shuffle_all_words, test_des.shuffle_nouns, test_des.shuffle_verb,
			# 					 test_des.shuffle_adj]
			perturb_functions = [test_des.shuffle_nouns_and_verb_adj, test_des.shuffle_allbut_nouns_verb_adj,
								 test_des.shuffle_all_words]
		else:
			perturb_functions = [test_des.shuffle_nouns_and_adj, test_des.shuffle_allbut_nouns_and_adj,
								 test_des.shuffle_within_trigrams, test_des.shuffle_trigrams, ]

		self.cla_name = [i.__name__ for i in perturb_functions]

		for img_id, ann in tqdm(enumerate(self.annotation), total=len(self.annotation), desc="Text Destroy Progress"):
			for i, caption in enumerate(ann['caption']):
				test_case = {}
				test_case["image"] = ann["image"]
				test_case["caption_options"] = [pre_caption(caption, max_words)]

				for perturb_fn in perturb_functions:
					test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))

				self.test_cases.append(test_case)

	def __len__(self):
		return len(self.test_cases)

	def __getitem__(self, index):
		test_case = self.test_cases[index]
		image_path = os.path.join(self.root_dir, test_case["image"])
		image = Image.open(image_path).convert('RGB')

		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
		return item

	def evaluate_scores(self, scores):
		if isinstance(scores, tuple):
			scores_i2t = scores[0]
			scores_t2i = scores[1].T  # Make it N_ims x N_text
		else:
			scores_t2i = scores
			scores_i2t = scores

		preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
		count = dict(Counter(preds))
		correct_mask = (preds == 0)
		result_records = []

		for ids, pr in count.items():
			if ids != 0:
				rela = self.cla_name[ids - 1]
			else:
				rela = "Correct"
			result_records.append({
				"Relation": rela,
				"Accuracy": pr / len(preds),
				"Count": pr,
				"Dataset": "Visual Genome Relation"
			})
		# result_records = [{"Precision@1": np.mean(correct_mask)}]
		return result_records


def get_visual_genome_attribute_ownership(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False,
										  *args, **kwargs):
	dataset = VG_Attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
							 image_perturb_fn=image_perturb_fn, download=download, attribute_ownership=True, *args,
							 **kwargs)
	return dataset


def get_visual_genome_sentence_logic(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False,
									 *args,
									 **kwargs):
	dataset = VG_Attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
							 image_perturb_fn=image_perturb_fn, download=download, logic=True, *args, **kwargs)
	return dataset


def get_visual_genome_spatial_relationship_bias(image_preprocess, text_perturb_fn=None, image_perturb_fn=None,
												download=False, *args, **kwargs):
	dataset = VG_Relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
						  image_perturb_fn=image_perturb_fn, download=download,
						  multi_spatial_relation=False, subordination_relation=True, *args, **kwargs)
	return dataset


def get_visual_genome_spatial_relationship_multirela(image_preprocess, text_perturb_fn=None, image_perturb_fn=None,
													 download=False, *args, **kwargs):
	dataset = VG_Relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
						  image_perturb_fn=image_perturb_fn, download=download, multi_spatial_relation=True, *args,
						  **kwargs)
	return dataset


def get_coco_semantic_structure(image_preprocess, max_words=30, download=False,
								root_dir=COCO_ROOT, split="test", *args, **kwargs):
	return COCO_Semantic_Structure(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
								   max_words=max_words, download=download)


def get_flickr30k_semantic_structure(image_preprocess, max_words=30, download=False,
									 root_dir=FLICKR_ROOT, split="test", *args, **kwargs):
	return Flickr30k_Semantic_Structure(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
										max_words=max_words,
										download=download)


####################
def get_visual_genome_relation(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False, *args,
							   **kwargs):
	dataset = VG_Relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
						  image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
	return dataset


def get_visual_genome_attribution(image_preprocess, text_perturb_fn=None, image_perturb_fn=None, download=False, *args,
								  **kwargs):
	dataset = VG_Attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
							 image_perturb_fn=image_perturb_fn, download=download)
	return dataset


def get_coco_order(image_preprocess, max_words=30, download=False,
				   root_dir=COCO_ROOT, split="test", *args, **kwargs):
	dataset = COCO_Semantic_Structure(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
									  max_words=max_words, download=download, semantic_structure=False)
	return dataset


def get_flickr30k_order(image_preprocess, max_words=30, download=False,
						root_dir=FLICKR_ROOT, split="test", *args, **kwargs):
	dataset = Flickr30k_Semantic_Structure(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
										   max_words=max_words, download=download, semantic_structure=False)
	return dataset
