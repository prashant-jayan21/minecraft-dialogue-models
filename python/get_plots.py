import os, argparse
from glob import glob
from plot_utils import plot_losses

def main(args):
	""" Gets all train log files for models in a given directory and writes the following to file:
	1. A plot of train loss vs epochs for each model in a .train.png file
	2. The corresponding data in a .train file
	1. A plot of validation loss vs epochs for each model in a .val.png file
	2. The corresponding data in a .val file
	"""
	# iterate over model directories that have successfully been trained and evaluated
	for config_file in glob(args.models_dir+'/**/config.txt', recursive=True):
		if not os.path.exists(config_file.replace('config.txt','eval.txt')):
			continue

		train_log_file = config_file.replace('config.txt', args.model + "_train.log")
		plot_losses(train_log_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=str, help='path for models saved on a specific date, each containing their own config & eval files')
	parser.add_argument('model', type=str, nargs='?', help='type of model trained')
	args = parser.parse_args()
	main(args)
