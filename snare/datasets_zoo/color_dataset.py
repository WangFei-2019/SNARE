import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFont
from collections import Counter

from torch.utils.data import Dataset
from matplotlib import colors
from .data_des import _colors
from .retrieval_dataset import pre_caption


class Colors_set(Dataset):
	def __init__(self, image_size, image_preprocess=None, max_multiple=0.1, prompt: str = None, extend_num=5):
		'''

		:param image_size:
		:param image_preprocess:
		:param max_multiple:
		:param prompt: example "a photo in {} color"
		'''
		self.image_preprocess = image_preprocess

		self.colors = _colors
		self.max_multiple = [1 - (max_multiple * i / extend_num) for i in range(extend_num + 1)]
		if isinstance(image_size, int):
			self.image_size = (image_size, image_size)
		elif isinstance(image_size, tuple or list):
			if len(image_size) == 1:
				self.image_size = (image_size[0], image_size[0])
			elif len(image_size) == 2:
				self.image_size = image_size
			else:
				raise NameError(
					f"The length of the \'image_size\', when which is a tulpe or list, "
					f"should be 1(side length), 2(length and width) or 3(length, width and channel), "
					f"or else it must be a int (side length).")
		else:
			raise TypeError
		self.blank_image = np.array(self.image_size)
		self.color_to_rgb()
		self.text = [pre_caption(self.id2text[i]) for i in range(len(self.id2text))] if prompt is None \
			else [pre_caption(prompt.format(self.id2text[i])) for i in range(len(self.id2text))]

	def color_to_rgb(self):
		self.image2id = dict()
		self.id2text = dict()
		# self.image_list = []
		id = 0
		for col in self.colors:
			try:
				rgb = colors.to_rgb(col)
			except:
				continue
			if tuple((np.array(rgb) * 255).astype("uint8")) in self.image2id.keys():
				continue
			elif col == "white":
				self.id2text[id] = col
				self.image2id[tuple((np.array(rgb) * 255).astype("uint8"))] = id
				id += 1
				continue
			self.id2text[id] = col
			for i in self.max_multiple:
				rgb_ = (np.array(rgb) * i * 255).astype("uint8")
				self.image2id[tuple(rgb_)] = id
			id += 1
		self.image_list = list(self.image2id.items())
		self.image_tru = list(self.image2id.values())


	def save_color_img(self, path="Samples.png", im_size=50):
		side_d = 5
		count = Counter(self.image_tru)
		ncols = max(count.values())
		nrows = len(count)
		pair = np.array(list(zip(count.keys(), count.values())))
		freq = list(pair[pair.argsort(0).T[1]].T[0])
		figure = np.ones(
			[ncols * im_size + (ncols - 1) * 2 + 2 * side_d + im_size // 2,
			 nrows * im_size + (nrows - 1) * 5 + 2 * side_d, 3]) * 255
		order = np.zeros([nrows], dtype=np.int)
		for rgb, ids in self.image_list:
			rgb = np.array(ImageOps.expand(Image.new('RGB', [im_size - 2, im_size - 2], rgb), border=1, fill=0))
			i = len(freq) - freq.index(ids) - 1
			figure[order[i] * (im_size + 2) + side_d: order[i] * (im_size + 2) + im_size + side_d,
			i * (im_size + 5) + side_d: i * (im_size + 5) + im_size + side_d, :] = rgb
			order[i] += 1
		figure = Image.fromarray(np.uint8(figure))
		draw = ImageDraw.Draw(figure)

		font_type = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/timesi.ttf", im_size * 2 // 5,
									   encoding="utf-8")
		for _ids in self.id2text.keys():
			i = len(freq) - freq.index(_ids) - 1
			anchor_bbox = draw.textbbox((0, 0), self.id2text[_ids], font=font_type, anchor='lt')
			anchor_center = (anchor_bbox[0] + anchor_bbox[2]) // 2, (anchor_bbox[1] + anchor_bbox[3]) // 2
			mask_bbox = font_type.getmask(self.id2text[_ids]).getbbox()
			mask_center = (mask_bbox[0] + mask_bbox[2]) // 2, (mask_bbox[1] + mask_bbox[3]) // 2
			draw.text([i * (im_size + 5) + side_d + im_size // 2 - (anchor_center[0] - mask_center[0]),
					   max(order) * im_size + side_d + im_size//2 -(anchor_center[1] - mask_center[1])],
					  text=self.id2text[_ids], fill=(0, 0, 0), font=font_type, anchor='mm')
		figure.save(path)
		print(f"All samples in the Color set is saved to {path}")

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		rgb, color_id = self.image_list[index]
		image = Image.new('RGB', self.image_size, rgb)
		if self.image_preprocess is not None:
			image = self.image_preprocess(image)

		return {"image": image, "idx": color_id}

	def evaluate_scores(self, scores):
		if isinstance(scores, tuple):
			scores = scores[0]
		classification = np.argmax(scores, axis=-1)
		cla_t = np.sum(self.image_tru == classification)
		acc = cla_t / len(self.image_tru)
		random_acc = 1 / (np.max(self.image_tru) + 1)

		print(
			f"*** Color classification ***\n"
			f"ACC: {acc:.3f}\n"
			f"Random ACC: {random_acc:.3f}\n"
			f"Categories: {self.id2text}\n"
			f"Sample Number: {len(self.image_tru)}\n")

		records = [{"acc": acc, "Random_acc": random_acc, "Categories": self.id2text,
					"Sample_num": len(self.image_tru)}]

		return records


def get_color_dataset(image_size, image_preprocess=None, max_multiple=0.3, prompt=None, extend_num=5):
	dataset = Colors_set(image_size=image_size, image_preprocess=image_preprocess, max_multiple=max_multiple,
						 prompt=prompt, extend_num=extend_num)
	return dataset
