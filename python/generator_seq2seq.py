import os, argparse
from utils import timestamp
from vocab import Vocabulary
import generate_seq2seq

# META SCRIPT THAT RUNS GENERATION AND EVALUATION FOR EACH MODEL IN A COLLECTION OF MODELS SEQUENTIALLY

def main(args):
	all_models_dirs = list()
	for root, dirs, files in os.walk(args.models_root_dir):
		if not dirs:
			all_models_dirs.append(root)

	model_specific_args = list(map(lambda x: (x, os.path.join(x, args.output_path) if args.output_path else None), all_models_dirs))
	print(model_specific_args)

	all_args = []
	for model_arg in model_specific_args:
		args_copy = argparse.Namespace(**vars(args))
		setattr(args_copy, "model_dir", model_arg[0])
		setattr(args_copy, "output_path", model_arg[1])
		all_args.append(args_copy)

	output = list(map(generate_seq2seq.main, all_args))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_root_dir', type=str, help='path for root dir containing all saved models obtained from grid search')
	parser.add_argument('--output_path', type=str, default=None, help='path for output file of generated sentences')
	parser.add_argument('--data_dir', type=str, default='../data/logs/', help='path for data jsons')
	parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/', help='path for gold config xmls')
	parser.add_argument('--saved_dataset_dir', type=str, default=None, help='path for saved dataset to use')
	parser.add_argument('--vocab_dir', type=str, default="../vocabulary/", help='path for vocabulary files')
	parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
	parser.add_argument('--seed', type=int, default=1234, help='random seed')
	parser.add_argument('--beam_size', type=int, default=10, help='beam size for beam search decoding')
	parser.add_argument('--max_decoding_length', type=int, default=50, help='max iterations for decoder when decoding w/o ground truth inputs') # FIXME: Do not use hard coded string
	parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")
	parser.add_argument('--decoding_strategy', type=str, default='beam', help='multinomial/beam')
	parser.add_argument('--gamma', type=float, default=0.8, help='gamma penalty')
	parser.add_argument('--regenerate_sentences', default=False, action='store_true', help='generate sentences for a model even if a generated sentences file already exists in its directory')
	parser.add_argument('--model_iteration', default='best', help='iteration of model to be evaluated: "best" or "final"')
	parser.add_argument('--split', default='val', help='data split from which sentences should be generated')
	parser.add_argument('--disable_shuffle', default=False, action='store_true', help='disable shuffling of the data to be generated from')
	args = parser.parse_args()
	print(timestamp(), args, "\n")
	main(args)