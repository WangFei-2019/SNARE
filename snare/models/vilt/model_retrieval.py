import torch
import torch.nn as nn
from .modules import vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transformers import BertTokenizer
from .modules import heads
from tqdm import tqdm


class ViLT_Retrieval(nn.Module):
	def __init__(self, config):
		super().__init__()
		bert_config = BertConfig(
			vocab_size=config["vocab_size"],
			hidden_size=config["hidden_size"],
			num_hidden_layers=config["num_layers"],
			num_attention_heads=config["num_heads"],
			intermediate_size=config["hidden_size"] * config["mlp_ratio"],
			max_position_embeddings=config["max_text_len"],
			hidden_dropout_prob=config["drop_rate"],
			attention_probs_dropout_prob=config["drop_rate"],
		)

		self.tokenizer = BertTokenizer.from_pretrained(
			config["tokenizer"], do_lower_case="uncased" in config["tokenizer"]
		)

		self.text_embeddings = BertEmbeddings(bert_config)
		self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])

		self.transformer = getattr(vit, config["vit"])(pretrained=False, config=config)

		self.pooler = heads.Pooler(config["hidden_size"])
		self.itm_score = heads.ITMHead(config["hidden_size"])
		self.config = config
		self.rank_output = nn.Linear(self.itm_score.fc.in_features, 1)


	def load_pretrained(self, ckpt_rpath):
		ckpt = torch.load(ckpt_rpath, map_location="cpu")
		state_dict = ckpt["state_dict"]
		msg = self.load_state_dict(state_dict, strict=False)
		print('load checkpoint from %s' % ckpt_rpath)
		print("missing_keys: ", msg.missing_keys)
		print("unexpected_keys: ", msg.unexpected_keys)
		self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
		self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]

	def text_embed(self, text):
		text_embed = self.text_embeddings(text.input_ids)
		text_type_embed = self.token_type_embeddings(torch.zeros_like(text.attention_mask))
		text_embed = text_embed + text_type_embed
		return text_embed, text.attention_mask, text.input_ids

	def vision_embed(self, image, image_token_type_idx=1):
		image_embed, image_masks, patch_index, image_labels = \
			self.transformer.visual_embed(image, max_image_len=self.config["max_image_len"])
		image_type_embed = self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
		image_embed = image_embed + image_type_embed
		return image_embed, image_masks

	def cross_encoder(self, image_embed, text_embed, image_mask, text_mask):
		co_embeds = torch.cat([text_embed, image_embed], dim=1)
		co_masks = torch.cat([text_mask, image_mask], dim=1)

		x = co_embeds

		for i, blk in enumerate(self.transformer.blocks):
			x, _attn = blk(x, mask=co_masks)

		x = self.transformer.norm(x)
		text_feats, image_feats = x[:, : text_embed.shape[1]], x[:, text_embed.shape[1]:],
		cls_feats = self.pooler(x)

		return text_feats, image_feats, cls_feats

	def forward(self, image, text):
		image_embed, image_mask = self.vision_embed(image)
		text_embed, text_mask, _ = self.text_embed(text)
		text_feats, image_feats, cls_feats = self.cross_encoder(image_embed, text_embed, image_mask, text_mask)

		return text_feats, image_feats, cls_feats
