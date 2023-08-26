import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

class BLIP2Model:
	def __init__(self, root_dir, device):
		# loads BLIP-2 pre-trained model
		self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl",
															 is_eval=True, device=device)
		self.device = device
		self.qustion = "are there {} in the image?"  # VG_Attribute_Ownership
		# self.qustion = "Is there \"{}\"?"  # VG_Subordination_Relationship, VG_Sentence_Logic
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
				image = group[0].unsqueeze(0).to(self.device)
				options = list(group[1:])
				score = []
				description = self.model.generate({"image": image, "prompt": "Question: Describe the image. Answer:"})[0]
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
					output = self.model.generate({"image": image, "prompt": f'Question: Describe the image. Answer: {description}. Question: {opt} Answer:'})[0].lower()
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
