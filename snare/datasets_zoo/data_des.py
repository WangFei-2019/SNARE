import numpy as np
import torch
import PIL
import random
import copy
import torch.nn.functional as nnf
from torchvision import transforms as T

from functools import partial
import mmcv
from mmdet.apis import inference_detector, init_detector
import spacy
import nltk


# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations
_pos = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "adj": ['JJ', 'JJR', 'JJS'],
		"verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"], }

_mask = "[MASK]"

_people = ['man', 'woman', 'boy', 'girl', 'people', 'men', "women"]

_people_reversal = {'man': "woman", 'woman': "man", 'boy': "girl", 'girl': "boy", 'people': "animals", 'men': "women",
					"women": "men"}

_colors = ['orange', 'green', 'red', 'white', 'black', 'pink', 'blue', 'purple', 'tan', 'grey', 'gray', 'yellow',
		   'gold', 'golden', 'dark', 'brown', 'silver']


class Img_Des():
	def __init__(self,
				 device='cuda:0',
				 config='/workspace/VLP_explore/mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py',
				 checkpoint='/workspace/VLP_explore/mmdetection/checkpoint/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037.pth'
				 ):
		'''
		:param device: 
		:param config: Choose to use a config and initialize the detector
		:param checkpoint: Setup a checkpoint file to load
		'''
		try:
			torch.multiprocessing.set_start_method('spawn')
		except:
			pass

		print("Prepare detection/segmentation model!")
		self.model = init_detector(config, checkpoint, device=device)  # or device='cuda:0'

	def change_images(self, img, detect_box=False, detect_segm=True, mask=False, score_thr=0.7, return_labels=False):
		return_Image = False
		return_tensor = False
		if isinstance(img, torch.Tensor):
			img = np.array(img.permute(1, 2, 0))
			return_tensor = True
		elif isinstance(img, PIL.Image.Image):
			img = np.array(img)
			return_Image = True
		# if not isinstance(img, np.ndarray):
		# 	img = np.array(img)
		bbox_result, segm_result = inference_detector(self.model, img)
		bboxes = np.vstack(bbox_result)

		labels = [np.full(bbox.shape[0], i, dtype=np.int32)
				  for i, bbox in enumerate(bbox_result)
				  ]
		labels = np.concatenate(labels)
		labels_ = None
		segms = None
		if segm_result is not None and len(labels) > 0:  # non empty
			segms = mmcv.concat_list(segm_result)
			if isinstance(segms[0], torch.Tensor):
				segms = torch.stack(segms, dim=0).detach().cpu().numpy()
			else:
				segms = np.stack(segms, axis=0)

		img = mmcv.imread(img).astype(np.uint8)

		if score_thr > 0:
			assert bboxes is not None and bboxes.shape[1] == 5
			scores = bboxes[:, -1]
			inds = scores > score_thr
			bboxes = bboxes[inds, :]
			labels = labels[inds]
			if segms is not None:
				segms = segms[inds, ...]

		# img = mmcv.bgr2rgb(img)

		width, height = img.shape[1], img.shape[0]
		# img = np.ascontiguousarray(img)
		# max_label = int(max(labels) if len(labels) > 0 else 0)

		num_bboxes = 0
		if bboxes is not None and detect_box:
			box_m = np.ones((height, width), dtype=np.uint8)
			num_bboxes = bboxes.shape[0]
			for i, bbox in enumerate(bboxes):
				bbox_int = bbox.astype(np.int32)
				box_m[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = 0

			img = img * box_m[:, :, None] if mask else img * (1 - box_m[:, :, None])
			labels_ = [self.model.CLASSES[i] for i in labels[:num_bboxes]]

		if segms is not None and detect_segm:
			segm_m = np.ones((height, width), dtype=np.uint8)
			if num_bboxes < segms.shape[0]:
				segms = segms[num_bboxes:]
				for m in segms:
					segm_m = segm_m * (m == False).astype(np.uint8)
			img = img * segm_m[:, :, None] if mask else img * (1 - segm_m[:, :, None])
			labels_ = [self.model.CLASSES[i] for i in labels]

		if return_Image:
			img = PIL.Image.fromarray(img)
		if return_tensor:
			img = torch.from_numpy(img).permute(1, 2, 0)
		if return_labels:
			return img, labels_
		else:
			return img

	def mask_region(self, x, n_rows=0.5):
		return self.change_images(x, detect_box=False, detect_segm=True, mask=True, score_thr=n_rows)

	def reserve_region(self, x, n_rows=0.5):
		return self.change_images(x, detect_box=False, detect_segm=True, mask=False, score_thr=n_rows)

	def shuffle_rows(self, x, n_rows=3):
		"""
		Shuffle the rows of the image tensor where each row has a size of 14 pixels.
		Tensor is of shape N x C x W x H
		"""
		x, return_image = _handle_image_4shuffle(x)
		patch_size = x.shape[-2] // n_rows
		u = nnf.unfold(x, kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
		# permute the patches of each image in the batch
		pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
		# fold the permuted patches back together
		f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)

		image = f.squeeze()  # C W H
		if return_image:
			return T.ToPILImage()(image.type(torch.uint8))
		else:
			return image

	def shuffle_columns(self, x, n_cols=3):
		"""
		Shuffle the columns of the image tensor where we'll have n_cols columns.
		Tensor is of shape N x C x W x H
		"""
		x, return_image = _handle_image_4shuffle(x)
		patch_size = x.shape[-1] // n_cols
		u = nnf.unfold(x, kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
		# permute the patches of each image in the batch
		pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
		# fold the permuted patches back together
		f = nnf.fold(pu, x.shape[-2:], kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
		image = f.squeeze()  # C W H
		if return_image:
			return T.ToPILImage()(image.type(torch.uint8))
		else:
			return image

	def shuffle_patches(self, x, n_ratio=4):
		"""
		Shuffle the rows of the image tensor where each row has a size of 14 pixels.
		Tensor is of shape N x C x W x H
		"""
		x, return_image = _handle_image_4shuffle(x)
		patch_size_x = x.shape[-2] // n_ratio
		patch_size_y = x.shape[-1] // n_ratio
		u = nnf.unfold(x, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
		# permute the patches of each image in the batch
		pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
		# fold the permuted patches back together
		f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y),
					 padding=0)
		image = f.squeeze()  # C W H
		if return_image:
			return T.ToPILImage()(image.type(torch.uint8))
		else:
			return image


class Text_Des:
	def __init__(self, emcoder="en_core_web_sm", shuffle=False):
		'''
		:param text_perturb_fn: the name of text destroy method.
		:param emcoder: 'en_core_web_sm' for efficiency / 'en_core_web_trf' for accuracy.
		:param is_random: shuffle the words or not.
		'''
		self.shuffle = shuffle

		print("Prepare SPACY NLP model!")
		self.nlp = spacy.load(emcoder)
		self.word_tokenize = nltk.word_tokenize

	def color_exchange(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		colors = copy.copy(_colors)
		extr_color = []
		color_idx = [i for i, token in enumerate(text) if token in _colors]
		for x in color_idx:
			colors.remove(text[x])
			extr_color.append(random.sample(colors, 1)[0])
			colors = copy.copy(_colors)
		extr_color = np.array(extr_color)
		text[color_idx] = extr_color
		return " ".join(text)

	def color_exchange_to_white(self, ex):
		'''
		exchange color to white, exchange white to black
		'''
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		color_idx = []
		white_idx = []
		for i, token in enumerate(text):
			if token in _colors and token != "white":
				color_idx.append(i)
			elif token == "white":
				white_idx.append(i)
		text[color_idx] = ["white" for i in range(len(color_idx))]
		text[white_idx] = ["black" for i in range(len(white_idx))]
		return " ".join(text)

	def color_mask(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		color_idx = [i for i, token in enumerate(text) if token in _colors]
		text[color_idx] = [_mask for i in range(len(color_idx))]
		return " ".join(text)

	def reserve_allbut_color(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		color_idx = [i for i, token in enumerate(text) if token not in _colors]
		return " ".join(text[color_idx])

	def entity_mask(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		entity_idx = [i for i, token in enumerate(text) if token in _people]
		text[entity_idx] = [_mask for i in range(len(entity_idx))]
		return " ".join(text)

	def entity_reversal(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		entity_idx = []
		entity = []
		for i, token in enumerate(text):
			if token in _people:
				entity_idx.append(i)
				entity.append(_people_reversal[token])
		text[entity_idx] = entity
		return " ".join(text)

	def reserve_noun(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		des_text = np.random.permutation(text[noun_idx]) if self.shuffle else text[noun_idx]
		return " ".join(des_text)

	def mask_noun(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		text[noun_idx] = [_mask for i in range(len(noun_idx))]
		return " ".join(text)

	def reserve_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		adj_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		des_text = np.random.permutation(text[adj_idx]) if self.shuffle else text[adj_idx]
		return " ".join(des_text)

	def mask_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		adj_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		text[adj_idx] = [_mask for i in range(len(adj_idx))]
		return " ".join(text)

	def reserve_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		des_text = np.random.permutation(text[verb_idx]) if self.shuffle else text[verb_idx]
		return " ".join(des_text)

	def mask_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		text[verb_idx] = [_mask for i in range(len(verb_idx))]
		return " ".join(text)

	def reserve_noun_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"] + _pos["adj"]]
		des_text = np.random.permutation(text[noun_adj_idx]) if self.shuffle else text[noun_adj_idx]
		return " ".join(des_text)

	def mask_noun_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"] + _pos["adj"]]
		text[noun_adj_idx] = [_mask for i in range(len(noun_adj_idx))]
		return " ".join(text)

	def reserve_noun_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"] + _pos["verb"]]
		des_text = np.random.permutation(text[noun_verb_idx]) if self.shuffle else text[noun_verb_idx]
		return " ".join(des_text)

	def mask_noun_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"] + _pos["verb"]]
		text[noun_verb_idx] = [_mask for i in range(len(noun_verb_idx))]
		return " ".join(text)

	def reserve_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		adj_verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"] + _pos["verb"]]
		des_text = np.random.permutation(text[adj_verb_idx]) if self.shuffle else text[adj_verb_idx]
		return " ".join(des_text)

	def mask_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		adj_verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"] + _pos["verb"]]
		text[adj_verb_idx] = [_mask for i in range(len(adj_verb_idx))]
		return " ".join(text)

	def reserve_noun_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_verb_idx = [i for i, token in enumerate(doc) if
							 token.tag_ in _pos["noun"] + _pos["adj"] + _pos["verb"]]
		des_text = np.random.permutation(text[noun_adj_verb_idx]) if self.shuffle else text[noun_adj_verb_idx]
		return " ".join(des_text)

	def mask_noun_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_verb_idx = [i for i, token in enumerate(doc) if
							 token.tag_ in _pos["noun"] + _pos["adj"] + _pos["verb"]]
		text[noun_adj_verb_idx] = [_mask for i in range(len(noun_adj_verb_idx))]
		return " ".join(text)

	def reserve_allbut_noun_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		wo_noun_adj_verb_idx = [i for i, token in enumerate(doc) if
								token.tag_ not in _pos["noun"] + _pos["adj"] + _pos["verb"]]
		des_text = np.random.permutation(text[wo_noun_adj_verb_idx]) if self.shuffle else text[wo_noun_adj_verb_idx]
		return " ".join(des_text)

	def shuffle_nouns(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		## Shuffle the nouns of the text
		text[noun_idx] = np.random.permutation(text[noun_idx])
		return " ".join(text)

	def shuffle_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		## Finding adjectives
		adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		## Shuffle the nouns of the text
		text[adjective_idx] = np.random.permutation(text[adjective_idx])
		return " ".join(text)

	def shuffle_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		## Shuffle the nouns of the text
		text[verb_idx] = np.random.permutation(text[verb_idx])
		return " ".join(text)

	def shuffle_nouns_and_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		## Finding adjectives
		adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		## Shuffle the nouns of the text
		text[noun_idx] = np.random.permutation(text[noun_idx])
		## Shuffle the adjectives of the text
		text[adjective_idx] = np.random.permutation(text[adjective_idx])
		return " ".join(text)

	def shuffle_nouns_and_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		## Finding adjectives
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		## Shuffle the nouns of the text
		text[noun_idx] = np.random.permutation(text[noun_idx])
		## Shuffle the adjectives of the text
		text[verb_idx] = np.random.permutation(text[verb_idx])
		return " ".join(text)

	def shuffle_nouns_and_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		## Shuffle the nouns of the text
		text[noun_idx] = np.random.permutation(text[noun_idx])
		## Shuffle the adjectives of the text
		text[verb_idx] = np.random.permutation(text[verb_idx])
		text[adjective_idx] = np.random.permutation(text[adjective_idx])
		return " ".join(text)

	def shuffle_all_words(self, ex):
		return " ".join(np.random.permutation(ex.split(" ")))

	def shuffle_allbut_nouns_and_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]+_pos["adj"]]
		## Finding adjectives
		else_idx = np.ones(text.shape[0])
		else_idx[noun_adj_idx] = 0

		else_idx = else_idx.astype(bool)
		## Shuffle everything that are nouns or adjectives
		text[else_idx] = np.random.permutation(text[else_idx])
		return " ".join(text)

	def shuffle_allbut_nouns_and_verb(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]+_pos["verb"]]
		## Finding adjectives
		else_idx = np.ones(text.shape[0])
		else_idx[noun_verb_idx] = 0

		else_idx = else_idx.astype(bool)
		## Shuffle everything that are nouns or adjectives
		text[else_idx] = np.random.permutation(text[else_idx])
		return " ".join(text)

	def shuffle_allbut_nouns_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_idx = [i for i, token in enumerate(doc) if
						token.tag_ in _pos["noun"] + _pos["adj"] + _pos["verb"]]
		## Finding adjectives

		else_idx = np.ones(text.shape[0])
		else_idx[noun_adj_idx] = 0

		else_idx = else_idx.astype(bool)
		## Shuffle everything that are nouns or adjectives
		text[else_idx] = np.random.permutation(text[else_idx])
		return " ".join(text)

	def shuffle_within_trigrams(self, ex):
		tokens = self.word_tokenize(ex)
		shuffled_ex = _trigram_shuffle(tokens)
		return shuffled_ex

	def shuffle_trigrams(self, ex):
		tokens = self.word_tokenize(ex)
		trigrams = _get_trigrams(tokens)
		random.shuffle(trigrams)
		shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
		return shuffled_ex


def get_text_perturb_fn(text_perturb_fn, shuffle=False):
	if text_perturb_fn is None:
		return None
	else:
		text_des = Text_Des(shuffle=shuffle)
		methods_dir = [i for i in (text_des.__dir__()) if "__" not in i]
		if text_perturb_fn in methods_dir:
			return getattr(text_des, text_perturb_fn)
		else:
			raise NameError(f"Unknown text perturbation function: {text_perturb_fn}."
							f"The methods you can choose is {methods_dir}")


def get_image_perturb_fn(image_perturb_fn, device="cpu"):
	if image_perturb_fn is None:
		return None
	else:
		img_des = Img_Des(device=device)
		methods_dir = [i for i in (img_des.__dir__()) if "__" not in i]
		image_perturb_fn = image_perturb_fn.split("-")
		if image_perturb_fn[0] in methods_dir:
			try:
				return partial(getattr(img_des, image_perturb_fn[0]), n_rows=int(image_perturb_fn[-1]))
			except:
				return getattr(img_des, image_perturb_fn[0])
		else:
			raise NameError(f"Unknown image perturbation function: {image_perturb_fn}."
							f"The methods you can choose is {methods_dir}")


# if image_perturb_fn == "shuffle_rows_4":
# 	return partial(shuffle_rows, n_rows=4)
# elif image_perturb_fn == "shuffle_patches_9":
# 	return partial(shuffle_patches, n_ratio=3)
# elif image_perturb_fn == "shuffle_cols_4":
# 	return partial(shuffle_columns, n_cols=4)
# elif image_perturb_fn is None:
# 	return None


def _get_trigrams(sentence):
	# Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
	trigrams = []
	trigram = []
	for i in range(len(sentence)):
		trigram.append(sentence[i])
		if i % 3 == 2:
			trigrams.append(trigram[:])
			trigram = []
	if trigram:
		trigrams.append(trigram)
	return trigrams


def _trigram_shuffle(sentence):
	trigrams = _get_trigrams(sentence)
	for trigram in trigrams:
		random.shuffle(trigram)
	return " ".join([" ".join(trigram) for trigram in trigrams])


def _handle_image_4shuffle(x):
	return_image = False
	if not isinstance(x, torch.Tensor):
		# print(f"x is not a tensor: {type(x)}. Trying to handle but fix this or I'll annoy you with this log")
		t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
		t = t.permute(0, 3, 1, 2)
		return_image = True
		return t, return_image
	if len(x.shape) != 4:
		# print("You did not send a tensor of shape NxCxWxH. Unsqueezing not but fix this or I'll annoy you with this log")
		return x.unsqueeze(dim=0), return_image
	else:
		# Good boi
		return x, return_image
