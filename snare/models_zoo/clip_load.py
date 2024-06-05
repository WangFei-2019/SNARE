import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


from . import clip


class CLIPModel:
	def __init__(self, model_name, device, root_dir):
		self.model, self.image_preprocess = clip.load(model_name, device=device, download_root=os.path.join(root_dir, "clip"))
		self.device = device

		# self.blank = self.model.encode_text(clip.tokenize("").to(self.device)).detach().cpu().numpy()

	@torch.no_grad()
	def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
		num_text = len(texts)
		text_embeds = []
		tqdm_loader = tqdm(range(0, num_text, text_batch_size))
		tqdm_loader.set_description("Computing text embeddings")
		for i in tqdm_loader:
			text = texts[i: min(num_text, i + text_batch_size)]
			text_input = clip.tokenize(text, truncate=True).to(self.device)
			text_feats = self.model.encode_text(text_input)
			if normalize:
				text_feats = F.normalize(text_feats, dim=-1)
			text_embeds.append(text_feats)

		text_embeds = torch.cat(text_embeds, dim=0)

		return text_embeds

	@torch.no_grad()
	def get_image_embeddings(self, image_loader, normalize=False):
		image_embeds = []
		tqdm_loader = tqdm(image_loader)
		tqdm_loader.set_description("Computing image embeddings")
		for batch in tqdm_loader:
			images = batch["image"]
			image_feats = self.model.encode_image(images.to(self.device))
			if normalize:
				image_feats = F.normalize(image_feats, dim=-1)
			image_embeds.append(image_feats)

		image_embeds = torch.cat(image_embeds, dim=0)
		return image_embeds

	@torch.no_grad()
	def get_retrieval_scores_dataset(self, loader):
		captions = loader.dataset.text
		text_embeds = self.get_text_embeddings(captions, normalize=True)
		image_embeds = self.get_image_embeddings(loader, normalize=True)
		scores = image_embeds @ text_embeds.T
		scores = scores.cpu().numpy()
		return scores

	@torch.no_grad()
	def get_retrieval_scores_batched(self, joint_loader):
		"""Computes the scores for each image_option / caption_option pair in the joint loader.

		Args:
			joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
			"image_options" is a list of images, and "caption_options" is a list of captions.

		Returns:
			all_scores: A numpy array containing the scores of the shape NxKxL,
			where N is the number of test cases, K is the number of image options per the test case,
			and L is the number of caption options per the test case.
		"""
		scores = []
		tqdm_loader = tqdm(joint_loader)
		tqdm_loader.set_description("Computing retrieval scores")
		for batch in tqdm_loader:
			image_options = []
			for i_option in batch["image_options"]:
				image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy()  # B x D
				image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # B x D
				image_options.append(np.expand_dims(image_embeddings, axis=1))

			caption_options = []
			for c_option in batch["caption_options"]:
				caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
				caption_embeddings = self.model.encode_text(caption_tokenized.to(self.device)).cpu().numpy()  # B x D

				# caption_embeddings = caption_embeddings - self.blank

				caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1,
																		 keepdims=True)  # B x D
				caption_options.append(np.expand_dims(caption_embeddings, axis=1))

			image_options = np.concatenate(image_options, axis=1)  # B x K x D
			caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
			batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L

			# # 例子矩阵
			# nkd = image_options
			# nld = caption_options
			#
			# # 广播矩阵形状
			# nkd_expanded = np.expand_dims(nkd, axis=2)  # shape: (3, 4, 1, 2)
			# nld_expanded = np.expand_dims(nld, axis=1)  # shape: (3, 1, 5, 2)
			#
			# # 计算差值
			# diff = nkd_expanded - nld_expanded  # shape: (3, 4, 5, 2)
			#
			# # 计算平方差并求和
			# squared_diff = diff ** 2  # shape: (3, 4, 5, 2)
			# sum_squared_diff = np.sum(squared_diff, axis=-1)  # shape: (3, 4, 5)
			#
			# # 计算欧式距离
			# batch_scores = np.sqrt(sum_squared_diff)  # shape: (3, 4, 5)
			#
			# batch_scores = batch_scores_cos / batch_scores

			scores.append(batch_scores)

		all_scores = np.concatenate(scores, axis=0)  # N x K x L
		return all_scores
