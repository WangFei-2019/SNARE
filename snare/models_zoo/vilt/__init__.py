import torch


def collate(batch):
	batch_size = len(batch)
	key = "image"
	img_sizes = [i[key].shape for i in batch]

	for size in img_sizes:
		assert (
				len(size) == 3
		), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

	max_height = max([i[1] for i in img_sizes])
	max_width = max([i[2] for i in img_sizes])

	new_images = torch.zeros(batch_size, 3, max_height, max_width)
	new_ids = torch.zeros(batch_size, dtype=torch.int16)
	for i in range(batch_size):
		img = batch[i][key]
		new_images[i, :, : img.shape[1], : img.shape[2]] = img
		new_ids[i] = batch[i]["idx"]
	dict_batch = {key: new_images, "idx": new_ids}
	return dict_batch
