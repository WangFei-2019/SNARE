import os
import torch
import yaml
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from .xvlm.model_retrieval import XVLM
from .xvlm.tokenization_bert import BertTokenizer
from .xvlm.tokenization_roberta import RobertaTokenizer
from snare.models.utils import MetricLogger

# All of the below URLs are taken from, and most of the implementation are heavily inspired from the wonderful https://github.com/salesforce/BLIP repo.

download_urls = {
	"model_url": "https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing",
	"vision_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_swinB_224.json",
	"config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/Pretrain_XVLM_base_4m.yaml",
	"bert_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_bert.json"
}


class XVLMModel:
	def __init__(self, weight_dir="weight", config_path="snare/models/xvlm/configs", device="cpu"):
		self.weight_dir = weight_dir
		self.config_path = os.path.join(config_path, "Pretrain_XVLM_base_4m.yaml")
		self.model_path = os.path.join(weight_dir, "xvlm", "4m_base_model_state_step_199999.th")
		self.bert_config_path = os.path.join(config_path, "config_bert.json")
		self.vision_config_path = os.path.join(config_path, "config_swinB_224.json")

		print(self.config_path)
		self.download(os.path.exists(self.config_path), os.path.exists(self.model_path))

		self.model = XVLM(self.config)

		self.model.load_pretrained(self.model_path, self.config, is_eval=True, is_pretrained=True)
		self.model = self.model.to(device)
		self.device = device
		self.config['k_test'] = 128 # 128 for f30k and 256 for coco

		if self.config['use_roberta']:
			self.tokenizer = RobertaTokenizer.from_pretrained(self.config['text_encoder'])
		else:
			self.tokenizer = BertTokenizer.from_pretrained(self.config['text_encoder'])

	def download(self, config_path_exist, model_path_exist):
		if not config_path_exist:
			os.makedirs(os.path.join(self.weight_dir, "configs"), exist_ok=True)
			print(f"Downloading X-VLM config to {self.weight_dir}/xvlm/configs...")
			config_url = download_urls["config_url"]
			self.config_path = os.path.join(self.weight_dir, "xvlm", "configs", "Pretrain_XVLM_base_4m.yaml")
			subprocess.call(["wget", "-c", config_url, "-O", self.bert_config_path])
		self.config = yaml.load(open(self.config_path, 'r'), Loader=yaml.Loader)
		if not model_path_exist:
			print(f"Downloading X-VLM Model to {self.weight_dir}/xvlm...")
			os.makedirs(os.path.join(self.weight_dir, 'xvlm'), exist_ok=True)
			model_url = download_urls["model_url"]
			subprocess.call(["wget", "-c", model_url, "-O", self.model_path])
		if not os.path.exists(self.config["vision_config"]):
			self.config["vision_config"] = os.path.join(self.weight_dir, "xvlm", "configs", "config_swinB_224.json")
			print(f"Downloading X-VLM vision config to {self.weight_dir}/xvlm/configs...")
			config_url = download_urls["vision_config_url"]
			subprocess.call(["wget", "-c", config_url, "-O", self.config["vision_config"]])
		if not os.path.exists(self.config["text_config"]):
			self.config["text_config"] = os.path.join(self.weight_dir, "xvlm", "configs", "config_swinB_224.json")
			print(f"Downloading bert config to {self.weight_dir}/xvlm/configs...")
			config_url = download_urls["bert_config_url"]
			subprocess.call(["wget", "-c", config_url, "-O", self.config["text_config"]])

	@torch.no_grad()
	def get_text_embeddings(self, texts, text_batch_size=256):
		num_text = len(texts)
		text_bs = 256
		text_ids = []
		text_embeds = []
		text_atts = []
		for i in tqdm(range(0, num_text, text_bs), desc="Computing text embeddings"):
			text = texts[i: min(num_text, i + text_bs)]
			text_input = self.tokenizer(text, padding='max_length', truncation=True,
										max_length=self.config["max_tokens"], return_tensors="pt").to(self.device)
			text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
												  mode='text')
			text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))
			text_embeds.append(text_embed)
			text_ids.append(text_input.input_ids)
			text_atts.append(text_input.attention_mask)

		text_embeds = torch.cat(text_embeds, dim=0)
		text_ids = torch.cat(text_ids, dim=0)
		text_atts = torch.cat(text_atts, dim=0)
		# text_ids[:,0] = self.tokenizer.enc_token_id
		return text_embeds, text_ids, text_atts

	@torch.no_grad()
	def get_image_embeddings(self, image_loader):
		image_feats = []
		image_embeds = []
		for batch in tqdm(image_loader, desc="Computing image embeddings"):
			image = batch["image"]
			image = image.to(self.device)
			image_feat = self.model.vision_encoder(image)
			image_embed = self.model.vision_proj(image_feat[:, 0, :])
			image_embed = F.normalize(image_embed, dim=-1)

			image_feats.append(image_feat.cpu())
			image_embeds.append(image_embed)

		image_feats = torch.cat(image_feats, dim=0)
		image_embeds = torch.cat(image_embeds, dim=0)
		return image_feats, image_embeds

	@torch.no_grad()
	def get_retrieval_scores_dataset(self, loader):
		texts = loader.dataset.text
		metric_logger = MetricLogger(delimiter="  ")
		text_embeds, text_ids, text_atts = self.get_text_embeddings(texts)
		image_feats, image_embeds = self.get_image_embeddings(loader)
		sims_matrix = image_embeds @ text_embeds.t()
		score_matrix_i2t = torch.full((image_embeds.shape[0], len(texts)), -100.0).to(self.device)

		num_tasks = 1
		rank = 0
		step = sims_matrix.size(0) // num_tasks + 1
		start = rank * step
		end = min(sims_matrix.size(0), start + step)

		for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation i2T")):
			topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)

			encoder_output = image_feats[start + i].repeat(self.config['k_test'], 1, 1).to(self.device)
			encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
			output = self.model.text_encoder(text_ids[topk_idx],
											 attention_mask=text_atts[topk_idx],
											 encoder_hidden_states=encoder_output,
											 encoder_attention_mask=encoder_att,
											 return_dict=True,
											 )
			score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
			score_matrix_i2t[start + i, topk_idx] = score + topk_sim

		sims_matrix = sims_matrix.t()
		score_matrix_t2i = torch.full((len(texts), image_feats.shape[0]), -100.0).to(self.device)

		step = sims_matrix.size(0) // num_tasks + 1
		start = rank * step
		end = min(sims_matrix.size(0), start + step)

		for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation T2i")):
			topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
			encoder_output = image_feats[topk_idx].to(self.device)
			encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
			output = self.model.text_encoder(text_ids[start + i].repeat(self.config['k_test'], 1),
											 attention_mask=text_atts[start + i].repeat(self.config['k_test'], 1),
											 encoder_hidden_states=encoder_output,
											 encoder_attention_mask=encoder_att,
											 return_dict=True,
											 )
			score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
			score_matrix_t2i[start + i, topk_idx] = score + topk_sim

		return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

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
				image_feat = self.model.vision_encoder(i_option.to(self.device))
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
				text_input = self.tokenizer(c_option, padding='max_length', truncation=True,
											max_length=self.config["max_tokens"], return_tensors="pt").to(self.device)
				text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
													  mode='text')
				text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

				text_embeds.append(text_embed.unsqueeze(1))
				text_ids.append(text_input.input_ids.unsqueeze(1))
				text_atts.append(text_input.attention_mask.unsqueeze(1))

			text_embeds = torch.cat(text_embeds, dim=1)
			text_ids = torch.cat(text_ids, dim=1)
			text_atts = torch.cat(text_atts, dim=1)

			s_i2t, s_t2i = self.run_scores_batched(image_embeds, image_feats, text_embeds, text_ids, text_atts)
			t2i_scores.append(s_t2i)
			i2t_scores.append(s_i2t)
		t2i_scores = np.concatenate(t2i_scores, axis=0)  # N x N_t x N_i
		t2i_scores = np.transpose(t2i_scores, (0, 2, 1))  # N x N_i x N_t
		i2t_scores = np.concatenate(i2t_scores, axis=0)  # N x N_i x N_t
		return t2i_scores, i2t_scores
