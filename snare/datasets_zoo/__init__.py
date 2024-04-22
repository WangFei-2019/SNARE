import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)

COCO_ROOT = f"{grandparent_directory}/dataset/data/coco"
FLICKR_ROOT = f"{grandparent_directory}/dataset/data/f30k"
CASSP_ROOT = f"{grandparent_directory}/dataset/data/prerelease_bow"


def get_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args,
				**kwargs):
	"""
	Helper function that returns a datasets_zoo object with an evaluation function.
	dataset_name: Name of the datasets_zoo.
	image_preprocess: Preprocessing function for images.
	text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
	image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
	download: Whether to allow downloading images if they are not found.
	"""
	# Classification Task
	if dataset_name == "Attribute_Ownership":
		from .snare_datasets import get_visual_genome_attribute_ownership
		return get_visual_genome_attribute_ownership(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
													 image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)

	if dataset_name == "Relationship_Composition":
		from .snare_datasets import get_visual_genome_spatial_relationship_bias
		return get_visual_genome_spatial_relationship_bias(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
														   image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
	elif dataset_name == "Spatial_Relationship":
		from .snare_datasets import get_visual_genome_spatial_relationship_multirela
		return get_visual_genome_spatial_relationship_multirela(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
																image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)


	elif dataset_name == "Negation_Logic":
		from .snare_datasets import get_visual_genome_sentence_logic
		return get_visual_genome_sentence_logic(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
												image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)


	elif dataset_name == "COCO_Semantic_Structure":
		from .snare_datasets import get_coco_semantic_structure
		return get_coco_semantic_structure(image_preprocess=image_preprocess, download=download, *args, **kwargs)
	elif dataset_name == "Flickr30k_Semantic_Structure":
		from .snare_datasets import get_flickr30k_semantic_structure
		return get_flickr30k_semantic_structure(image_preprocess=image_preprocess, download=download, *args, **kwargs)

	#  Classification Task in ARO
	elif dataset_name == "VG_Attribution":
		from .snare_datasets import get_visual_genome_attribution
		return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
													 image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
	elif dataset_name == "VG_Relation":
		from .snare_datasets import get_visual_genome_relation
		return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
										  image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
	elif dataset_name == "COCO_Order":
		from .snare_datasets import get_coco_order
		return get_coco_order(image_preprocess=image_preprocess, download=download, *args, **kwargs)
	elif dataset_name == "Flickr30k_Order":
		from .snare_datasets import get_flickr30k_order
		return get_flickr30k_order(image_preprocess=image_preprocess, download=download, *args, **kwargs)

	# Retrieval Task
	elif dataset_name == "COCO":
		from .retrieval_dataset import get_coco_retrieval
		return get_coco_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
								  image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
	elif dataset_name == "Flickr30k":
		from .retrieval_dataset import get_flickr30k_retrieval
		return get_flickr30k_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
									   image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)

	# Classification Task On Color
	elif dataset_name == "Color":
		from .color_dataset import get_color_dataset
		return get_color_dataset(image_preprocess=image_preprocess, *args, **kwargs)
	else:
		raise ValueError(f"Unknown datasets_zoo {dataset_name}")
