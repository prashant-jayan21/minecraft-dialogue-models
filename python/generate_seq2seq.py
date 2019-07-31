import os, argparse, torch, pickle, pprint, json
from glob import glob
from utils import *
from vocab import Vocabulary
from data_loader import CwCDataset
from train_and_eval import generate, multinomial_generate, multinomial_generate_seq2seq
from bleu import compute_bleu
from argparse import Namespace

def main(args):
	initialize_rngs(args.seed)

	if not args.development_mode:
		torch.multiprocessing.set_sharing_strategy('file_system')

	config_file = os.path.join(args.model_dir, "config.txt")
	_, config_params = get_config_params(config_file)
	print(config_params)

	output_path = args.output_path
	args_sfx = None

	if output_path is None:
		args_sfx = '-'+args.model_iteration+'-'+args.split+('' if not args.development_mode else '-development_mode')
		args_sfx += '-multinomial' if args.decoding_strategy == 'multinomial' else '-beam_'+str(args.beam_size)+('-gamma_'+str(args.gamma) if args.gamma else '')
		output_path = os.path.join(args.model_dir, 'generated_sentences'+args_sfx+'.txt')

	if not args.regenerate_sentences and (os.path.isfile(output_path) or os.path.isfile(os.path.join(args.model_dir, 'fail.txt'))):
		print("\nGenerated sentences already exist for model", args.model_dir+"; skipping.\n")
		return

	eval_files = glob(args.model_dir+'/eval-*.txt')
	if len(eval_files) == 0:
		print("Model", args.model_dir, "has not finished training; skipping.")
		return

	print("\n"+timestamp(), "Generated sentences will be written to", print_dir(output_path, 6), "...\n")

	model_type = config_params["model"]

	with open(config_params["encoder_vocab_path"], 'rb') as f:
		encoder_vocab = pickle.load(f)

	with open(config_params["decoder_vocab_path"], 'rb') as f:
		decoder_vocab = pickle.load(f)

	model_files = glob(args.model_dir+"/*-"+args.model_iteration+".pkl")

	models = {}
	for model_file in model_files:
		with open(model_file, 'rb') as f:
			if not torch.cuda.is_available():
				model = torch.load(f, map_location="cpu")
			else:
				model = torch.load(f)
			if "flatten_parameters" in dir(model):
				model.flatten_parameters() # TODO: flatten for all sub-modules recursively
			if "encoder" in model_file:
				models["encoder"] = model
			elif "decoder" in model_file:
				models["decoder"] = model

	lower_enc = "lower" in os.path.abspath(config_params["encoder_vocab_path"])
	lower_dec = "lower" in os.path.abspath(config_params["decoder_vocab_path"])

	if lower_dec != lower_enc:
		print("Encoder and decoder vocabs have to be cased the same way. Different casing is currently not supported.")
		sys.exit(0)

	lower = lower_dec

	saved_dataset_dir = config_params['saved_dataset_dir'] if args.saved_dataset_dir is None else args.saved_dataset_dir

	# load test or val set
	test_dataset = CwCDataset(
		model=model_type, split=args.split, lower=lower,
		data_dir=args.data_dir, gold_configs_dir=args.gold_configs_dir, saved_dataset_dir=saved_dataset_dir, vocab_dir=args.vocab_dir,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, load_dataset=True, transform=None
	)

	test_dataset.set_args(num_prev_utterances=config_params["num_prev_utterances"], blocks_max_weight=config_params["blocks_max_weight"], use_builder_actions=config_params['use_builder_actions'], include_empty_channel=config_params['include_empty_channel'], use_condensed_action_repr=config_params['use_condensed_action_repr'], action_type_sensitive=config_params['action_type_sensitive'], feasible_next_placements=config_params['feasible_next_placements'],  spatial_info_window_size=config_params["spatial_info_window_size"], counters_extra_feasibility_check=config_params["counters_extra_feasibility_check"], use_existing_blocks_counter=config_params["use_existing_blocks_counter"])
	test_dl = test_dataset.get_data_loader(batch_size=1, shuffle=not args.disable_shuffle, num_workers=args.num_workers)

	print(models)

	init_args = Namespace(set_decoder_hidden=config_params['set_decoder_hidden'],
						  concatenate_decoder_inputs=config_params['concatenate_decoder_inputs'],
						  concatenate_decoder_hidden=config_params['concatenate_decoder_hidden'],
						  decoder_input_concat_size=config_params['decoder_input_concat_size'],
						  decoder_hidden_concat_size=config_params['decoder_hidden_concat_size'],
						  advance_decoder_t0=config_params['advance_decoder_t0'])

	try:
		to_print = None
		if args.decoding_strategy == "multinomial":
			# sampling according to the distribution
			generated_utterances = multinomial_generate_seq2seq(
				models["encoder"], models["decoder"],
				config_params["init_decoder_with_encoder"], test_dl, decoder_vocab,
				beam_size=args.beam_size, max_length=args.max_decoding_length,
				development_mode=args.development_mode
			)
		elif args.decoding_strategy == "beam":
			# beam search decoding
			generated_utterances, to_print = generate(
				models["encoder"], models["decoder"],
				test_dl, decoder_vocab,
				beam_size=args.beam_size, max_length=args.max_decoding_length, args=init_args,
				development_mode=args.development_mode, gamma=args.gamma
			)

		# compute some stats
		lengths = list(map(lambda x: list(map(lambda y: len(y), x["generated_utterance"])), generated_utterances))
		lengths = [j for i in lengths for j in i]
		mean_length = np.mean(lengths)
		std_dev_length = np.std(lengths)

		print("Stats:")
		print("Mean utterance length:", mean_length)
		print("Standard deviation:", std_dev_length)

		def wordify_ground_truth_utterance(ground_truth_utterance):
			"""
				Maps from a 2d tensor to a list of tokens and removes eos symbol
			"""
			return list(map(lambda x: list(map(lambda y: decoder_vocab.idx2word[y.item()], x)), ground_truth_utterance))[0][:-1]

		# compute BLEU scores
		reference_corpus = list(map(lambda x: [wordify_ground_truth_utterance(x["ground_truth_utterance"])], generated_utterances))
		translation_corpus = list(map(lambda x: x["generated_utterance"][0], generated_utterances))

		bleu_1_results = compute_bleu(reference_corpus, translation_corpus, max_order=1, smooth=False)
		bleu_2_results = compute_bleu(reference_corpus, translation_corpus, max_order=2, smooth=False)
		bleu_3_results = compute_bleu(reference_corpus, translation_corpus, max_order=3, smooth=False)
		bleu_4_results = compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False)

		print("Bleu-1:")
		print(bleu_1_results[0])
		print("Bleu-2:")
		print(bleu_2_results[0])
		print("Bleu-3:")
		print(bleu_3_results[0])
		print("Bleu-4:")
		print(bleu_4_results[0])

		# better formatting
		def format_prev_utterances(prev_utterances):
			prev_utterances = list(map(lambda x: list(map(lambda y: encoder_vocab.idx2word[y.item()], x)), prev_utterances))
			prev_utterances = list(map(lambda x: " ".join(x), prev_utterances))[0]
			return prev_utterances

		def format_ground_truth_utterance(ground_truth_utterance):
			ground_truth_utterance = wordify_ground_truth_utterance(ground_truth_utterance)
			ground_truth_utterance = " ".join(ground_truth_utterance)
			return ground_truth_utterance

		def format_generated_utterance(generated_utterance):
			generated_utterance = list(map(lambda x: " ".join(x), generated_utterance))
			return generated_utterance

		def format_block_counters(block_counters):
			t2i = sorted(type2id.keys())
			formatted = {}
			for key in block_counters:
				formatted[key] = {}
				value_list = block_counters[key].tolist()
				for i in range(len(t2i)):
					formatted[key][t2i[i]] = value_list[0][0][i]
			return formatted

		# if model_type == "seq2seq_attn":
		# 	def format(output_obj):
		# 		prev_utterances = format_prev_utterances(output_obj["prev_utterances"])
		# 		ground_truth_utterance = format_ground_truth_utterance(output_obj["ground_truth_utterance"])
		# 		generated_utterance = format_generated_utterance(output_obj["generated_utterance"])

		# 		return {
		# 			"prev_utterances": prev_utterances,
		# 			"ground_truth_utterance": ground_truth_utterance,
		# 			"generated_utterance": generated_utterance
		# 		}

		if model_type == "seq2seq_world_state":
			def format(output_obj):
				ground_truth_utterance = format_ground_truth_utterance(output_obj["ground_truth_utterance"])
				generated_utterance = format_generated_utterance(output_obj["generated_utterance"])

				return {
					"ground_truth_utterance": ground_truth_utterance,
					"generated_utterance": generated_utterance,
					"built_config": None,
					"gold_config": None
				}

		elif model_type == "world_state_next_actions":
			def format(output_obj):
				ground_truth_utterance = format_ground_truth_utterance(output_obj["ground_truth_utterance"])
				generated_utterance = format_generated_utterance(output_obj["generated_utterance"])

				return {
					"ground_truth_utterance": ground_truth_utterance,
					"generated_utterance": generated_utterance,
					"next_actions_raw": output_obj["next_actions_raw"]
				}

		elif model_type == "utterances_and_next_actions" or model_type == "cnn_3d":
			def format(output_obj):
				prev_utterances = format_prev_utterances(output_obj["prev_utterances"])
				ground_truth_utterance = format_ground_truth_utterance(output_obj["ground_truth_utterance"])
				generated_utterance = format_generated_utterance(output_obj["generated_utterance"])

				return {
					"prev_utterances": prev_utterances,
					"ground_truth_utterance": ground_truth_utterance,
					"generated_utterance": generated_utterance,
					"next_actions_raw": output_obj["next_actions_raw"],
					"gold_next_actions_raw": output_obj["gold_next_actions_raw"]
				}

		elif model_type == "utterances_and_block_counters" or model_type == "utterances_and_block_region_counters" or model_type == "seq2seq_attn":
			def format(output_obj):
				prev_utterances = format_prev_utterances(output_obj["prev_utterances"])
				ground_truth_utterance = format_ground_truth_utterance(output_obj["ground_truth_utterance"])
				neg_sample_utterance = format_ground_truth_utterance(output_obj["neg_sample_utterance"]) if output_obj["neg_sample_utterance"] else None
				generated_utterance = format_generated_utterance(output_obj["generated_utterance"])
				block_counters = format_block_counters(output_obj["block_counters"])

				return {
					"prev_utterances": prev_utterances,
					"ground_truth_utterance": ground_truth_utterance,
					"neg_sample_utterance": neg_sample_utterance,
					"generated_utterance": generated_utterance,
					"next_actions_raw": output_obj["next_actions_raw"],
					"gold_next_actions_raw": output_obj["gold_next_actions_raw"],
					"block_counters": block_counters,
					"json_id": output_obj["json_id"],
					"sample_id": output_obj["sample_id"],
					"ground_truth_utterance_raw": output_obj["ground_truth_utterance_raw"]
				}

		generated_utterances = map(format, generated_utterances)

		generated_utterances_formatted = json.dumps(list(generated_utterances), indent=4, separators= (",\n", ": "), sort_keys=True)

		print("\n"+timestamp(), "Writing generated sentences to", print_dir(output_path, 6), "...\n")

		with open(output_path, 'w') as f:
			f.write("Stats:\n")
			f.write("Mean utterance length: " + str(mean_length) + "\n")
			f.write("Standard deviation: " + str(std_dev_length) + "\n")
			f.write("Bleu-1:\n")
			f.write(str(bleu_1_results[0]) + "\n")
			f.write("Bleu-2:\n")
			f.write(str(bleu_2_results[0]) + "\n")
			f.write("Bleu-3:\n")
			f.write(str(bleu_3_results[0]) + "\n")
			f.write("Bleu-4:\n")
			f.write(str(bleu_4_results[0]) + "\n")
			f.write("\n")
			f.write(generated_utterances_formatted)

		with open(os.path.join(args.model_dir, 'mul-std-bleu'+args_sfx+'.csv'), 'w') as f:
			f.write(str(mean_length)+','+str(std_dev_length)+','+str(bleu_1_results[0])+','+str(bleu_2_results[0])+','+str(bleu_3_results[0])+','+str(bleu_4_results[0]))

		if to_print:
			with open(os.path.join(args.model_dir, 'raw_sentences'+args_sfx+'.txt'), 'w') as f:
				for line in to_print:
					f.write(str(line)+'\n')

	except KeyboardInterrupt:
		print("Quitting generation early.")
		with open(os.path.join(args.model_dir, 'fail.txt'), 'w') as f:
			f.write('Generation stopped early.')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_dir', type=str, help='path for saved model to generate from')
	parser.add_argument('--data_dir', type=str, default='../data/logs/', help='path for data jsons')
	parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/', help='path for gold config xmls')
	parser.add_argument('--saved_dataset_dir', type=str, default=None, help='path for saved dataset to use')
	parser.add_argument('--vocab_dir', type=str, default="../vocabulary/", help='path for vocabulary files')
	parser.add_argument('--output_path', type=str, default=None, help='path for output file of generated sentences')
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
