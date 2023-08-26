import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from snare.models import get_model
from snare.datasets_zoo import get_dataset
from snare import set_seed, _default_collate, save_scores
from snare.models.vilt import collate
from snare.datasets_zoo.data_des import get_text_perturb_fn, get_image_perturb_fn
from snare import datasets_zoo


def config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--data_path", default="/workspace/dataset/data", type=str)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--num_workers", default=4, type=int)
	parser.add_argument("--model_name", default="vilt", choices=["flava", "x-vlm", "clip", "blip"], type=str)
	parser.add_argument("--dataset", default="Flickr30k", type=str, choices=["Flickr30k", "COCO"])
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--text_perturb_fn", default=None, type=str,
						help="Perturbation function to apply to the text.")
	parser.add_argument("--image_perturb_fn", default=None, type=str,
						help="Perturbation function to apply to the images.")

	parser.add_argument("--download", action="store_true",
						help="Download the datasets_zoo if it doesn't exist. (Default: False)")
	parser.add_argument("--save_scores", action="store_false",
						help="Save the scores for the retrieval. (Default: True)")
	parser.add_argument("--output_dir", default="./outputs", type=str)
	parser.add_argument("--extra_info", default=None, type=str)
	return parser.parse_args()


def main(args):
	set_seed(args.seed)
	datasets_zoo.COCO_ROOT = args.data_path
	datasets_zoo.FLICKR_ROOT = args.data_path

	model, image_preprocess = get_model(args.model_name, args.device, root_dir="weight")
	text_perturb_fn = get_text_perturb_fn(args.text_perturb_fn)
	image_perturb_fn = get_image_perturb_fn(args.image_perturb_fn, device=args.device)

	dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn,
						  image_perturb_fn=image_perturb_fn, download=args.download)
	# For some models we just pass the PIL images, so we'll need to handle them in the collate_fn.
	collate_fn = _default_collate if image_preprocess is None else None

	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
						collate_fn=collate_fn)

	scores = model.get_retrieval_scores_dataset(loader)
	result_records = dataset.evaluate_scores(scores)

	for record in result_records:
		record.update(
			{"Model": args.model_name, "Dataset": args.dataset, "Text Perturbation Strategy": args.text_perturb_fn,
			 "Seed": args.seed, "Image Perturbation Strategy": args.image_perturb_fn, "extra_info": args.extra_info})

	df = pd.DataFrame(result_records)
	output_file = os.path.join(args.output_dir, f"{args.dataset}.csv")
	os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None
	print(f"Saving results to {output_file}")
	if os.path.exists(output_file):
		all_df = pd.read_csv(output_file, index_col=0)
		all_df = pd.concat([all_df, df])
		all_df.to_csv(output_file)

	else:
		df.to_csv(output_file)

	if args.save_scores:
		save_scores(scores, args)


if __name__ == "__main__":
	args = config()
	main(args)
