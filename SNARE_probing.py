import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from snare.models import get_model
from snare.datasets_zoo import data_des, get_dataset
from snare import set_seed, _default_collate, save_scores, datasets_zoo
from snare.models.vilt import collate


def config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--data_path", default="/root/wf/dataset", type=str)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--num_workers", default=4, type=int)
	parser.add_argument("--model_name", default="llava", choices=["blip2", "llava", "flava", "x-vlm", "clip", "blip", "vilt"],
						type=str)
	parser.add_argument("--dataset", default="COCO_Semantic_Structure", type=str,
						choices=["VG_Attribute_Ownership", "VG_Subordination_Relationship",
								 "VG_Spatial_Relationship", "VG_Sentence_Logic",
								 "COCO_Semantic_Structure", "Flickr30k_Semantic_Structure",
								 "VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"])

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
	# datasets_zoo.COCO_ROOT = os.path.join(args.data_path, args.dataset)
	# datasets_zoo.FLICKR_ROOT = os.path.join(args.data_path, args.dataset)
	datasets_zoo.COCO_ROOT = os.path.join(args.data_path, "coco")
	datasets_zoo.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
	datasets_zoo.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")

	model, image_preprocess = get_model(args.model_name, args.device, root_dir="weight")

	dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

	# For some models we just pass the PIL images, so we'll need to handle them in the collate_fn.
	collate_fn = _default_collate if image_preprocess is None else None

	joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
							  collate_fn=collate_fn)

	scores = model.get_retrieval_scores_batched(joint_loader)
	if args.model_name in ['llava', 'blip2']:
		result_records = dataset.evaluate_vllm_scores(scores)
	else:
		result_records = dataset.evaluate_scores(scores)

	for record in result_records:
		record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed})
	if args.extra_info is None:
		output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}.csv")
	else:
		output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}_{args.extra_info}.csv")
	df = pd.DataFrame(result_records)
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
