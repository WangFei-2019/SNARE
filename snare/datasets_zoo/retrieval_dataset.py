import os
import re
import json
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from . import COCO_ROOT, FLICKR_ROOT
from .utils import AverageMeter


def pre_caption(caption, max_words=50):
	caption = re.sub(
		r"([.!\"()*#:;~])",
		' ',
		caption.lower(),
	)
	caption = re.sub(
		r"\s{2,}",
		' ',
		caption,
	)
	caption = caption.rstrip('\n')
	caption = caption.strip(' ')

	# truncate caption
	caption_words = caption.split(' ')
	if len(caption_words) > max_words:
		caption = ' '.join(caption_words[:max_words])

	return caption


class COCO_Retrieval(Dataset):
	def __init__(self, image_preprocess=None, root_dir=COCO_ROOT, max_words=30, split="test",
				 image_perturb_fn=None, text_perturb_fn=None, download=False):
		"""
		COCO Retrieval Dataset.
		image_preprocess: image preprocessing function
		root_dir: The directory of the coco dataset. This directory should contain test2014 files.
		max_words: Cropping the caption to max_words.
		split: 'val' or 'test'
		image_perturb_fn: image perturbation function for patch permutation experiments.
		download: Whether to download the dataset if it does not exist.
		"""
		self.root_dir = root_dir
		if not os.path.exists(root_dir):
			print("Directory for COCO could not be found!")
			if download:
				print(f"If you want to download COCO dataset to {COCO_ROOT}? (Enter \"y\")")
				if input() == "y":
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
		self.image_perturb_fn = image_perturb_fn
		self.image_root = root_dir

		self.text = []
		self.image = []
		self.txt2img = {}
		self.img2txt = {}

		txt_id = 0
		for img_id, ann in enumerate(self.annotation):
			self.image.append(ann['image'])
			self.img2txt[img_id] = []
			for i, caption in enumerate(ann['caption']):
				self.text.append(pre_caption(caption, max_words))
				self.img2txt[img_id].append(txt_id)
				self.txt2img[txt_id] = img_id
				txt_id += 1

		if text_perturb_fn is not None:
			self.text = [text_perturb_fn(t) for t in
						 tqdm(self.text, total=len(self.text), desc="Text Destroy Precess")]

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self, index):
		image_path = os.path.join(self.image_root, self.annotation[index]['image'])
		image = Image.open(image_path).convert('RGB')

		if self.image_perturb_fn is not None:
			image = self.image_perturb_fn(image)

		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		return {"image": image, "idx": index}

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

		print(f"COCO results across {scores_i2t.shape} samples. ")
		prec_at_1 = AverageMeter()
		prec_at_5 = AverageMeter()
		prec_at_10 = AverageMeter()

		rank_i2t = []
		rank_t2i = []
		img_len, text_len = scores_i2t.shape

		# Text retrieval
		tqdm_iterator = tqdm(range(len(self.img2txt)))
		for i in tqdm_iterator:
			top10_captions = np.argsort(scores_i2t[i])[-10:]
			# top5_captions = np.argsort(scores_i2t[i])[-5:]
			true_captions = self.img2txt[i]
			rank_i2t.append(
				np.min([text_len - np.where(np.argsort(scores_i2t[i]) == j_)[0][0] for j_ in true_captions]) - 1)

			prec_at_1.update(len(set(true_captions) & set(top10_captions[-1:])) > 0)
			prec_at_5.update(len(set(true_captions) & set(top10_captions[-5:])) > 0)
			prec_at_10.update(len(set(true_captions) & set(top10_captions)) > 0)

			tqdm_iterator.set_description(
				f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f},"
				f" Prec@10: {prec_at_10.avg:.3f}")

		# Image Retrieval
		image_prec_at_1 = AverageMeter()
		image_prec_at_5 = AverageMeter()
		image_prec_at_10 = AverageMeter()

		tqdm_iterator = tqdm(range(len(self.txt2img)))
		for i in tqdm_iterator:
			top10_images = np.argsort(scores_t2i[:, i])[-10:]
			# top5_images = np.argsort(scores_t2i[:, i])[-5:]
			true_image = self.txt2img[i]
			rank_t2i.append(img_len - np.where(np.argsort(scores_t2i[:, i]) == true_image)[0][0] - 1)

			image_prec_at_1.update(true_image in top10_images[-1:])
			image_prec_at_5.update(true_image in top10_images[-5:])
			image_prec_at_10.update(true_image in top10_images)

			tqdm_iterator.set_description(
				f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f},"
				f" Prec@10: {image_prec_at_10.avg:.3f}")
		medr_i2t = np.floor(np.median(rank_i2t)) + 1
		medr_t2i = np.floor(np.median(rank_t2i)) + 1
		meanr_i2t = np.array(rank_i2t).mean()
		meanr_t2i = np.array(rank_t2i).mean()
		ar_i2t = (image_prec_at_1.avg + image_prec_at_5.avg + image_prec_at_10.avg) / 3
		ar_t2i = (prec_at_1.avg + prec_at_5.avg + prec_at_10.avg) / 3

		print(
			f"Text Retrieval \n Average Recall: {ar_t2i:.3f}, MedianR(medr): {medr_t2i}, MeanR(meanr): {meanr_t2i:.3f}\n"
			f"Image Retrieval \n Average Recall: {ar_i2t:.3f}, MedianR(medr): {medr_i2t}, MeanR(meanr): {meanr_i2t:.3f}")

		records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg,
					"ImagePrec@10": image_prec_at_10.avg,
					"medr_i2t": medr_i2t, "meanr_i2t": meanr_i2t, "rank_i2t": rank_i2t,
					"TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg,
					"TextPrec@10": prec_at_10.avg,
					"medr_t2i": medr_t2i, "meanr_t2i": meanr_t2i, "rankt2i": rank_t2i}]

		return records


class Flickr30k_Retrieval(Dataset):
	def __init__(self, image_preprocess, split, root_dir=FLICKR_ROOT, max_words=30,
				 image_perturb_fn=None, text_perturb_fn=None, *args, **kwargs):
		'''
		Flickr30k dataset for retrieval.
		image_preprocess: image preprocessing function
		root_dir: The directory of the coco dataset. This directory should contain test2014 files.
		max_words: Cropping the caption to max_words.
		split: 'val' or 'test'
		image_perturb_fn: image perturbation function for patch permutation experiments.
		download: Whether to download the dataset if it does not exist.
		'''
		urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
				'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
		filenames = {'val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}

		if not os.path.exists(root_dir):
			print("Directory for Flickr30k could not be found!")
			flickr_url = "https://forms.illinois.edu/sec/229675"
			raise RuntimeError(
				f"You need to manually sign up and download the dataset "
				f"from {flickr_url} and place it in the `root_dir` ({root_dir}).")

		download_url(urls[split], root_dir)

		self.annotation = json.load(open(os.path.join(root_dir, filenames[split]), 'r'))
		self.image_preprocess = image_preprocess
		self.image_perturb_fn = image_perturb_fn
		self.root_dir = root_dir

		self.text = []
		self.image = []
		self.txt2img = {}
		self.img2txt = {}

		txt_id = 0
		for img_id, ann in enumerate(self.annotation):
			self.image.append(ann['image'])
			self.img2txt[img_id] = []
			for i, caption in enumerate(ann['caption']):
				self.text.append(pre_caption(caption, max_words))
				self.img2txt[img_id].append(txt_id)
				self.txt2img[txt_id] = img_id
				txt_id += 1

		if text_perturb_fn is not None:
			self.text = [text_perturb_fn(t) for t in
						 tqdm(self.text, total=len(self.text), desc="Text Destroy Precess")]

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self, index):
		image_path = os.path.join(self.root_dir, self.annotation[index]['image'])
		image = Image.open(image_path).convert('RGB')

		if self.image_perturb_fn is not None:
			image = self.image_perturb_fn(image)

		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		return {"image": image, "idx": index}

	def evaluate_scores(self, scores):
		if isinstance(scores, tuple):
			scores_i2t = scores[0]
			scores_t2i = scores[1].T  # Make it N_ims x N_text

		else:
			scores_t2i = scores
			scores_i2t = scores

		print(f"Flickr30k Retrieval results across {scores_i2t.shape} samples. ")
		prec_at_1 = AverageMeter()
		prec_at_5 = AverageMeter()
		prec_at_10 = AverageMeter()

		rank_i2t = []
		rank_t2i = []
		img_len, text_len = scores_i2t.shape

		# Text retrieval
		tqdm_iterator = tqdm(range(len(self.img2txt)))
		for i in tqdm_iterator:
			top10_captions = np.argsort(scores_i2t[i])[-10:]
			# top5_captions = np.argsort(scores_i2t[i])[-5:]
			true_captions = self.img2txt[i]
			rank_i2t.append(
				np.min([text_len - np.where(np.argsort(scores_i2t[i]) == j_)[0][0] for j_ in true_captions]) - 1)

			prec_at_1.update(len(set(true_captions) & set(top10_captions[-1:])) > 0)
			prec_at_5.update(len(set(true_captions) & set(top10_captions[-5:])) > 0)
			prec_at_10.update(len(set(true_captions) & set(top10_captions)) > 0)

			tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f},"
										  f" Prec@10: {prec_at_10.avg:.3f}")

		# Image Retrieval
		image_prec_at_1 = AverageMeter()
		image_prec_at_5 = AverageMeter()
		image_prec_at_10 = AverageMeter()

		tqdm_iterator = tqdm(range(len(self.txt2img)))
		for i in tqdm_iterator:
			top10_images = np.argsort(scores_t2i[:, i])[-10:]
			# top5_images = np.argsort(scores_t2i[:, i])[-5:]
			true_image = self.txt2img[i]
			rank_t2i.append(img_len - np.where(np.argsort(scores_t2i[:, i]) == true_image)[0][0] - 1)

			image_prec_at_1.update(true_image in top10_images[-1:])
			image_prec_at_5.update(true_image in top10_images[-5:])
			image_prec_at_10.update(true_image in top10_images)

			tqdm_iterator.set_description(
				f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f},"
				f" Prec@10: {image_prec_at_10.avg:.3f}")

		medr_i2t = np.floor(np.median(rank_i2t)) + 1
		medr_t2i = np.floor(np.median(rank_t2i)) + 1
		meanr_i2t = np.array(rank_i2t).mean() + 1
		meanr_t2i = np.array(rank_t2i).mean() + 1
		ar_i2t = (image_prec_at_1.avg + image_prec_at_5.avg + image_prec_at_10.avg) / 3
		ar_t2i = (prec_at_1.avg + prec_at_5.avg + prec_at_10.avg) / 3

		print(
			f"Text Retrieval \n Average Recall: {ar_t2i:.3f}, MedianR(medr): {medr_t2i}, MeanR(meanr): {meanr_t2i:.3f}\n"
			f"Image Retrieval \n Average Recall: {ar_i2t:.3f}, MedianR(medr): {medr_i2t}, MeanR(meanr): {meanr_i2t:.3f}")

		records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg,
					"ImagePrec@10": image_prec_at_10.avg,
					"medr_i2t": medr_i2t, "meanr_i2t": meanr_i2t, "rank_i2t": rank_i2t,
					"TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg,
					"TextPrec@10": prec_at_10.avg,
					"medr_t2i": medr_t2i, "meanr_t2i": meanr_t2i, "rankt2i": rank_t2i}]
		return records

	def download(self):
		raise NotImplementedError("Flickr30k dataset is not available for download. Please "
								  "find it from \"https://forms.illinois.edu/sec/229675\".")


def get_coco_retrieval(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False,
					   root_dir=COCO_ROOT, split="test"):
	dataset = COCO_Retrieval(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
							 image_perturb_fn=image_perturb_fn, text_perturb_fn=text_perturb_fn, max_words=max_words,
							 download=download)
	return dataset


def get_flickr30k_retrieval(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False,
							root_dir=FLICKR_ROOT, split="test"):
	dataset = Flickr30k_Retrieval(root_dir=root_dir, split=split, image_preprocess=image_preprocess,
								  image_perturb_fn=image_perturb_fn, text_perturb_fn=text_perturb_fn,
								  max_words=max_words,
								  download=download)
	return dataset
