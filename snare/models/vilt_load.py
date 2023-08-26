import os
import yaml
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch

from .vilt.model_retrieval import ViLT_Retrieval
from snare.models.utils import MetricLogger
from .vilt.modules.dist_utils import all_gather

# All of the below URLs are taken from, and most of the implementation are heavily inspired from the wonderful https://github.com/salesforce/BLIP repo.

download_urls = {
	"model_url": "https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",
	"config_url": "will load to our project"
}


class ViLTModel:
	def __init__(self, weight_dir="weight", config_path="snare/models/vilt/configs", device="cpu"):
		self.weight_dir = weight_dir
		self.model_path = os.path.join(weight_dir, "vilt", "vilt_200k_mlm_itm.ckpt")
		self.config_path = os.path.join(config_path, "vilt_200k_mlm_itm.yaml")

		self.download(os.path.exists(self.config_path), os.path.exists(self.model_path))

		config = yaml.load(open(self.config_path, 'r'), Loader=yaml.Loader)
		config["load_path"] = self.model_path

		model = ViLT_Retrieval(config=config)
		model.load_pretrained(self.model_path)
		self.model = model.to(device)
		self.device = device
		self.config = config
		self.config['k_test'] = 128  # 128 for f30k and 256 for coc

	# self.config['k_test'] = 128  # 128 for f30k and 256 for coco

	def download(self, config_path_exist, model_path_exist):
		if not config_path_exist:
			os.makedirs(os.path.join(self.weight_dir, "configs"), exist_ok=True)
			print(f"Downloading ViLT config to {self.weight_dir}/vilt/configs...")
			config_url = download_urls["config_url"]
			self.config_path = os.path.join(self.weight_dir, "blip", "configs", "vilt_200k_mlm_itm.yaml")
			subprocess.call(["wget", "-c", config_url, "-O", self.config_path])
		if not model_path_exist:
			print(f"Downloading ViLT Model to {self.weight_dir}...")
			model_url = download_urls["model_url"]
			subprocess.call(["wget", "-c", model_url, "-O", self.model_path])

	@torch.no_grad()
	def get_text_embeddings(self, texts, text_batch_size=256):
		num_text = len(texts)
		text_bs = 256
		text_embeds = []
		text_atts = []
		for i in tqdm(range(0, num_text, text_bs), desc="Computing text embeddings"):
			text = texts[i: min(num_text, i + text_bs)]
			text = self.model.tokenizer(text, padding='max_length', truncation=True,
										max_length=self.config["max_text_len"],
										return_tensors="pt").to(self.device)
			text_embed, text_att, _ = self.model.text_embed(text)
			text_embeds.append(text_embed)
			text_atts.append(text_att)

		text_embeds = torch.cat(text_embeds, dim=0)
		text_atts = torch.cat(text_atts, dim=0)
		# text_ids[:, 0] = self.model.tokenizer.eos_token_id
		return text_embeds, text_atts

	@torch.no_grad()
	def get_image_embeddings(self, image_loader):
		image_embeds = []
		image_masks = []
		for batch in tqdm(image_loader, desc="Computing image embeddings"):
			image = batch["image"]
			image = image.to(self.device)
			image_embed, image_mask = self.model.vision_embed(image)
			image_embeds.append(image_embed)
			image_masks.append(image_mask)

		image_embeds = torch.cat(image_embeds, dim=0)
		image_masks = torch.cat(image_masks, dim=0)
		return image_embeds, image_masks

	@torch.no_grad()
	def get_retrieval_scores_dataset(self, loader):
		texts = loader.dataset.text
		image_bs = 32
		text_bs = 32
		text_embeds, text_masks = self.get_text_embeddings(texts)
		image_embeds, image_masks = self.get_image_embeddings(loader)
		from torch.utils.data import TensorDataset, DataLoader
		text_ = TensorDataset(text_embeds, text_masks)
		image_ = TensorDataset(image_embeds, image_masks)
		text_preload = DataLoader(text_, batch_size=text_bs)
		image_preload = DataLoader(image_, batch_size=image_bs)
		rank_scores = np.zeros((len(image_embeds), len(text_embeds)))
		i = 0
		pbar = tqdm(image_preload, desc=f"rank loop → 0/{len(text_embeds)}", total=int(len(image_embeds)/image_bs))
		for image_embed, image_mask in pbar:
			image_l = len(image_embed)
			image_embed_ = image_embed.unsqueeze(1).repeat(1, text_bs, 1, 1)
			image_embed_ = image_embed_.reshape(-1, image_embed.shape[-2], image_embed.shape[-1])
			image_mask_ = image_mask.unsqueeze(1).repeat(1, text_bs, 1)
			image_mask_ = image_mask_.reshape(-1, image_mask.shape[-1])
			j = 0
			for text_embed, text_mask in text_preload:
				text_l = len(text_embed)
				if text_l != text_bs:
					image_embed_ = image_embed.unsqueeze(1).repeat(1, text_l, 1, 1)
					image_embed_ = image_embed_.reshape(-1, image_embed.shape[-2], image_embed.shape[-1])
					image_mask_ = image_mask.unsqueeze(1).repeat(1, text_l, 1)
					image_mask_ = image_mask_.reshape(-1, image_mask.shape[-1])
				pbar.set_description(f"rank loop → {j}/{len(text_embeds)}")
				text_embed = text_embed.repeat(image_l, 1, 1)
				text_mask = text_mask.repeat(image_l, 1)
				score = self.model.rank_output(
					self.model.cross_encoder(image_embed_, text_embed, image_mask_, text_mask)[2])
				score = score.reshape(image_l, text_l)
				rank_scores[i:i + image_l, j: j + text_l] = score.cpu()
				j += text_l
		return rank_scores

	def run_scores_batched(self, image_embeds, image_feats, text_embeds, text_ids, text_atts):
		# Should return something with shape (n_tests, n_image_options, n_text_options)
		# Image embeds and all: (n_tests, n_image_options, embed_dim)
		# Text embeds and all: (n_tests, n_text_options, embed_dim)

		# Score matrix should be of the size: (n_tests, n_image_options, n_text_options)

		sims_matrix = torch.einsum('ijk,ilk->ijl', image_embeds,
								   text_embeds)  # (n_tests, n_image_options, n_text_options)

		score_matrix_i2t = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(
			self.device)

		for i, sims in enumerate(sims_matrix):
			for j in range(sims.shape[0]):
				encoder_output = image_feats[i, j].repeat(sims_matrix.shape[2], 1, 1).to(self.device)
				encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
				output = self.model.text_encoder(text_ids[i],
												 attention_mask=text_atts[i],
												 encoder_hidden_states=encoder_output,
												 encoder_attention_mask=encoder_att,
												 return_dict=True)
				score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
				score_matrix_i2t[i, j] = score + sims[j]

		sims_matrix = sims_matrix.permute(0, 2, 1)  # (n_tests, n_text_options, n_image_options)
		score_matrix_t2i = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(
			self.device)

		for i, sims in enumerate(sims_matrix):
			for j in range(sims.shape[0]):
				encoder_output = image_feats[i].to(self.device)
				encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
				output = self.model.text_encoder(text_ids[i, j].repeat(sims_matrix.shape[2], 1),
												 attention_mask=text_atts[i, j].repeat(sims_matrix.shape[2], 1),
												 encoder_hidden_states=encoder_output,
												 encoder_attention_mask=encoder_att,
												 return_dict=True)
				score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
				score_matrix_t2i[i, j] = score + sims[j]

		return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

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
		t2i_scores, i2t_scores = [], []
		for batch in tqdm(joint_loader):
			image_feats = []
			image_embeds = []
			for i_option in batch["image_options"]:
				image_feat = self.model.visual_encoder(i_option.to(self.device))
				image_embed = self.model.vision_proj(image_feat[:, 0, :])  # B x D
				image_embed = F.normalize(image_embed, dim=-1)

				image_feats.append(image_feat.unsqueeze(1))
				image_embeds.append(image_embed.unsqueeze(1))

			image_feats = torch.cat(image_feats, dim=1)
			image_embeds = torch.cat(image_embeds, dim=1)

			text_ids = []
			text_embeds = []
			text_atts = []

			for c_option in batch["caption_options"]:
				c_option = list(c_option)
				text_input = self.model.tokenizer(c_option, padding='max_length', truncation=True, max_length=35,
												  return_tensors="pt").to(self.device)
				text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
													  mode='text')
				text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

				text_embeds.append(text_embed.unsqueeze(1))
				text_ids.append(text_input.input_ids.unsqueeze(1))
				text_atts.append(text_input.attention_mask.unsqueeze(1))

			text_embeds = torch.cat(text_embeds, dim=1)
			text_ids = torch.cat(text_ids, dim=1)
			text_atts = torch.cat(text_atts, dim=1)
			text_ids[:, :, 0] = self.model.tokenizer.enc_token_id

			s_i2t, s_t2i = self.run_scores_batched(image_embeds, image_feats, text_embeds, text_ids, text_atts)
			t2i_scores.append(s_t2i)
			i2t_scores.append(s_i2t)

		t2i_scores = np.concatenate(t2i_scores, axis=0)  # N x N_t x N_i
		t2i_scores = np.transpose(t2i_scores, (0, 2, 1))  # N x N_i x N_t
		i2t_scores = np.concatenate(i2t_scores, axis=0)  # N x N_i x N_t
		print(t2i_scores.shape, i2t_scores.shape)
		return t2i_scores, i2t_scores
