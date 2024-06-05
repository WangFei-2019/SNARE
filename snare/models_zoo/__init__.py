from PIL import Image
from torchvision import transforms
import torch

def get_model(model_name, device, root_dir="~/.cache", dataset_name="f30k"):
	"""
	Helper function that returns a model and a potential image preprocessing function.
	"""
	if model_name == "clip":
		from .clip_load import CLIPModel
		clip_model = CLIPModel('ViT-B/32', device=device, root_dir=root_dir)
		image_preprocess = clip_model.image_preprocess
		return clip_model, image_preprocess

	elif model_name == "blip":
		from .blip_load import BLIPModel
		blip_model = BLIPModel(root_dir=root_dir, device=device)
		image_preprocess = transforms.Compose([
			transforms.Resize((blip_model.config['image_size'], blip_model.config['image_size']),
							  interpolation=transforms.functional.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
		if dataset_name == "f30k":
			blip_model.config["k_test"] = 128
		elif dataset_name == "coco":
			blip_model.config["k_test"] = 256
		return blip_model, image_preprocess

	elif model_name == "x-vlm":
		from .xvlm_load import XVLMModel
		xvlm_model = XVLMModel(root_dir=root_dir, device=device)
		image_preprocess = transforms.Compose([
			transforms.Resize((xvlm_model.config['image_res'], xvlm_model.config['image_res']),
							  interpolation=Image.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])
		if dataset_name == "f30k":
			xvlm_model.config["k_test"] = 128
		elif dataset_name == "coco":
			xvlm_model.config["k_test"] = 256
		return xvlm_model, image_preprocess

	elif model_name == "flava":
		from .flava_load import FlavaModel
		flava_model = FlavaModel(root_dir=root_dir, device=device)
		image_preprocess = None
		return flava_model, image_preprocess

	elif model_name == "llava":
		from .llava_load import LLaVAModel
		from transformers import CLIPImageProcessor
		llava_model = LLaVAModel(root_dir, device=device)
		image_preprocess = CLIPImageProcessor.from_pretrained(llava_model.model.config.mm_vision_tower,
															  torch_dtype=torch.float16)
		return llava_model, image_preprocess

	elif model_name == "blip2":
		from .blip2_load import BLIP2Model
		blip2_model = BLIP2Model(root_dir, device=device)
		# image_preprocess = blip2_model.vis_processors 
		return blip2_model, None

	elif model_name == "minigpt4":
		pass

	# elif model_name == "vilt":
	# 	from .vilt_load import ViLTModel
	# 	from .vilt.transforms import keys_to_transforms
	# 	vilt_model = ViLTModel(root_dir=root_dir, device=device)
	# 	image_preprocess = \
	# 	keys_to_transforms(vilt_model.config["test_transform_keys"], size=vilt_model.config["image_size"])[0]
	# 	if dataset_name == "f30k":
	# 		vilt_model.config["k_test"] = 128
	# 	elif dataset_name == "coco":
	# 		vilt_model.config["k_test"] = 256
	# 	return vilt_model, image_preprocess

	else:
		raise ValueError(f"Unknown model {model_name}")

