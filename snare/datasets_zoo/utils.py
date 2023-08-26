import numpy as np

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def top_n_accuracy(predictions, targets, n):
	assert len(predictions) == len(targets), "Number of predictions and targets must match."
	correct_count = 0
	total = len(targets)

	# 注释掉下面两句时，概率相同时优先考虑最后的类别。否则优先考虑最前面的类别
	predictions = predictions[:, ::-1]
	targets = targets[:, ::-1]

	sorted_indexes = np.argsort(predictions, axis=1)[:, ::-1]
	for i in range(total):
		if np.where(targets[i] == 1) in sorted_indexes[i][:n]:
			correct_count += 1

	accuracy = correct_count / total

	return accuracy, correct_count
