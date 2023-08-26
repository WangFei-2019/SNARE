import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from snare.models import get_model
from snare.datasets_zoo import data_des, get_dataset
from snare import set_seed, _default_collate, save_scores
from snare.models.vilt import collate
from snare.datasets_zoo.data_des import get_text_perturb_fn, get_image_perturb_fn
from snare import datasets_zoo


def config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--num_workers", default=4, type=int)
	parser.add_argument("--model_name", default="clip", choices=["flava", "x-vlm", "clip", "blip"], type=str)
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--prompt", default=None, type=str, nargs="+", help="For example: a photo in {} color. a {} photo. the photo color is {}.")
	parser.add_argument("--max_multiple", default=0.5, type=float, help="The max multiple in color image extension.")
	parser.add_argument("--extend_num", default=5, type=int, help="The number of extended images in color image extension.")
	parser.add_argument("--save_scores", action="store_false",
						help="Save the scores for the retrieval. (Default: True)")
	parser.add_argument("--output_dir", default="./outputs", type=str)
	parser.add_argument("--extra_info", default=None, type=str)
	return parser.parse_args()


def main(args):
	set_seed(args.seed)
	args.prompt = " ".join(args.prompt) if args.prompt is not None else None

	model, image_preprocess = get_model(args.model_name, args.device, root_dir="weight")
	model.model.eval()

	dataset = get_dataset("Color", image_size=224, image_preprocess=image_preprocess,
						  max_multiple=args.max_multiple, extend_num=args.extend_num, prompt=args.prompt)
	if not os.path.exists(f"{args.output_dir}/colorset_{args.max_multiple}_{args.extend_num}.png"):
		dataset.save_color_img(path=f"{args.output_dir}/colorset_{args.max_multiple}_{args.extend_num}.png", im_size=50)
	try:
		model.config['k_test'] = max(dataset.image_tru) + 1
	except:
		pass
	# For some models we just pass the PIL images, so we'll need to handle them in the collate_fn.
	collate_fn = _default_collate if image_preprocess is None else None

	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
						collate_fn=collate_fn)

	scores = model.get_retrieval_scores_dataset(loader)
	result_records = dataset.evaluate_scores(scores)

	for record in result_records:
		record.update(
			{"Model": args.model_name, "Dataset": args.dataset, "text prompt": args.prompt,
			 "Seed": args.seed, "extra_info": args.extra_info,
			 "max_multiple": args.max_multiple, "extend_numextend_num": args.extend_num})

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
