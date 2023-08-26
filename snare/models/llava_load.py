import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from llava import LlavaLlamaForCausalLM
from transformers import CLIPImageProcessor, AutoTokenizer, StoppingCriteria
from llava.conversation import conv_templates
import copy

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disable_torch_init():
	"""
	Disable the redundant torch default initialization to accelerate model creation.
	"""
	import torch
	setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
	setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class LLaVAModel:
	def __init__(self, root_dir, device):
		self.model = LlavaLlamaForCausalLM.from_pretrained("/root/wf/SNARE_VLP_Knowledge_Probing/weight/LLaVA-7B-v0",
														   torch_dtype=torch.float16).cuda(device=device)
		self.tokenizer = AutoTokenizer.from_pretrained("/root/wf/SNARE_VLP_Knowledge_Probing/weight/LLaVA-7B-v0")
		mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
		self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
		if mm_use_im_start_end:
			self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

		vision_tower = self.model.model.vision_tower[0]
		vision_tower.to(device=device, dtype=torch.float16)
		vision_config = vision_tower.config
		vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
		vision_config.use_im_start_end = mm_use_im_start_end
		if mm_use_im_start_end:
			vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids(
				[DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
		image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

		self.qustion = "Are there {} in the image?"  # VG_Attribute_Ownership
		# self.qustion = "Is the description, \"{}\", right?"  # VG_Subordination_Relationship
		print(self.qustion)

		if mm_use_im_start_end:
			self.qs_ = "Describe the image." + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
		else:
			self.qs_ = "Describe the image." + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

		self.conv = conv_templates["simple"].copy()
		self.device = device

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
		for batch in tqdm(joint_loader):
			batch_scores = []
			for group in zip(batch['image_options'][0]['pixel_values'][0], *batch['caption_options']):
				image = group[0].unsqueeze(0).to(self.device)
				options = list(group[1:])

				conv = copy.deepcopy(self.conv)
				conv.append_message(conv.roles[0], self.qs_)
				prompt = conv.get_prompt()

				inputs = self.tokenizer([prompt])
				input_ids = torch.as_tensor(inputs.input_ids).cuda(self.device)
				keywords = ['###']
				stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

				with torch.inference_mode():
					description_ids = self.model.generate(
						input_ids,
						images=image.half(),
						do_sample=False,
						temperature=0.7,
						max_new_tokens=1024,
						stopping_criteria=[stopping_criteria])
				input_token_len = input_ids.shape[1]
				description = self.tokenizer.batch_decode(description_ids[:, input_token_len:], skip_special_tokens=True)[0]
				score = []
				for opt in options:
					for opt in options:
						if "is" in opt:
							opt = opt.replace(" is", "")
							opt = "is " + opt + "in the image?"
							opt.replace("and", "and is")
						elif "are" in opt:
							opt = opt.replace(" are", "")
							opt = "are " + opt + "in the image?"
						else:
							opt = self.qustion.format(opt)

					prompt_ask = prompt + description + f'Human: {opt}###'
					inputs = self.tokenizer([prompt_ask])
					input_ids = torch.as_tensor(inputs.input_ids).cuda(self.device)
					stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

					with torch.inference_mode():
						output_ids = self.model.generate(
							input_ids,
							images=image.half(),
							do_sample=False,
							temperature=0.7,
							max_new_tokens=1024,
							stopping_criteria=[stopping_criteria])
					input_token_len = input_ids.shape[1]
					output = \
						self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].lower()
					if "yes" in output:
						score.append(1)
					elif "no" in output:
						score.append(0)
					else:
						score.append(0)
						print(f"There are not \"Yes\" or \"No\" in answer. \n The answer is: {output}")
				batch_scores.append(score)
			scores.append(batch_scores)
		all_scores = np.concatenate(scores, axis=0)  # N x K x L
		return all_scores


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
	def __init__(self, keywords, tokenizer, input_ids):
		self.keywords = keywords
		self.tokenizer = tokenizer
		self.start_len = None
		self.input_ids = input_ids

	def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		if self.start_len is None:
			self.start_len = self.input_ids.shape[1]
		else:
			outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
			for keyword in self.keywords:
				if keyword in outputs:
					return True
		return False
