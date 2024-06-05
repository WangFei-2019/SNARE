import os
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

class BLIP2Model:
	def __init__(self, root_dir, device):
		self.processors = AutoProcessor.from_pretrained(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl") if os.path.exists(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl")) else "Salesforce/blip2-flan-t5-xl",)
		self.model = AutoModelForVisualQuestionAnswering.from_pretrained(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl") if os.path.exists(os.path.join(root_dir, "Salesforce/blip2-flan-t5-xl")) else "Salesforce/blip2-flan-t5-xl", device_map=device, torch_dtype=torch.float16, cache_dir=root_dir)

		self.model.eval()

		self.device = device
		self.qustion = "are there {} in the image?" 
		print(self.qustion)

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
			for group in zip(batch['image_options'][0], *batch['caption_options']):
				inputs = self.processors(images=group[0], text="Question: Describe the image. Answer:", return_tensors="pt").to(device=self.device, dtype=torch.bfloat16)
				options = list(group[1:])
				score = []
				description = self.model.generate(**inputs, max_length=100)
				description = self.processors.batch_decode(description, skip_special_tokens=True)[0].strip()
				for opt in options:
					if "is" in opt:
						opt = opt.replace(" is", "")
						opt = "is " + opt + " in the image?"
						opt.replace("and", "and is")
					elif "are" in opt:
						opt = opt.replace(" are", "")
						opt = "are " + opt + " in the image?"
					else:
						opt = self.qustion.format(opt)
					inputs = self.processors(images=group[0], text=f'Question: Describe the image. Answer: {description}. Question: {opt} Answer:', return_tensors="pt").to(device=self.device, dtype=torch.bfloat16)
					description = self.model.generate(**inputs, max_length=50)
					output = self.processors.batch_decode(description, skip_special_tokens=True)[0].strip().lower()
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
