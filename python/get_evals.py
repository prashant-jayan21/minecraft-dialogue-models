import sys, os, argparse, numpy as np, evaluation_analysis, json
from glob import glob
from utils import get_config_params, Logger
from collections import defaultdict

def append_eval_values(eval_statistics, param, value, eval_values):
	if not eval_statistics.get(param):
		eval_statistics[param] = {}

	if not eval_statistics[param].get(value):
		eval_statistics[param][value] = defaultdict(list)

	for eval_value in eval_values:
		eval_statistics[param][value][eval_value].append(eval_values[eval_value])

def main(args):
	""" Gets all eval.txt files for models in a given directory and compiles them into a single document (as well as single csv).
	Also prints information on the best-performing model. """
	eval_writer = open(os.path.join(args.models_dir, 'cumulative_evals-'+args.model_iteration+'.txt'), 'w')
	eval_csv_writer = open(os.path.join(args.models_dir, 'cumulative_evals-'+args.model_iteration+'.csv'), 'w')

	csv_header_written = False
	csv_header = 'saved model path, best epoch, validation loss, validation perplexity, '

	sys.stdout = Logger(os.path.join(args.models_dir, 'cumulative_evals-'+args.model_iteration+'.log'))

	best_validation_loss = None
	best_model_path = None
	best_config = None
	best_eval = None

	validation_losses = []
	perplexities = []
	epochs = []

	all_params = {}

	modified_eval_types = ['simple_terms', 'colors', 'spatial_relations', 'dialogue', 'synonym']
	agreement_types = ['next_placements', 'all_placements', 'all_removals']

	eval_statistics = {}
	eval_terms = [x.strip() for x in args.eval_terms.split(',')]

	for config_file in glob(args.models_dir+'/**/config.txt', recursive=True):
		_, config_params = get_config_params(config_file)
		for param in config_params:
			value = config_params[param]
			if config_params[param] is None:
				value = "None"
			all_params[param] = type(value)

	# iterate over model directories that have successfully been trained and evaluated
	for config_file in glob(args.models_dir+'/**/config.txt', recursive=True):
		print('Accumulating evals for model:', '/'.join(config_file.split('/')[:-1]))
		if not os.path.exists(config_file.replace('config.txt','eval-'+args.model_iteration+'.txt')):
			continue

		model_path = ('/'.join(os.path.abspath(config_file).split("/")[:-1]))
		content = '='*89+'\nsaved model path: '+model_path+'\n'
		csv = model_path+','

		# get the configuration of parameters used for this particular model
		config_content, config_params = get_config_params(config_file)

		with open(config_file.replace('config.txt','eval-'+args.model_iteration+'.txt'), 'r') as f:
			eval_content = f.read()

		content += '\n'+config_content+'\n'+eval_content
		eval_writer.write(content+'\n')

		# parse the model's eval.txt file
		eval_values = {}
		for line in eval_content.split('\n'):
			if line.startswith('Best model found at') or line.startswith('Final model found at'):
				epoch = int(line.split()[-1].replace('.',''))
				eval_values["epoch"] = epoch
				epochs.append(epoch)

			elif line.startswith('Loss:'):
				validation_loss = float(line.split()[-1])
				eval_values["validation_loss"] = validation_loss

				if not best_validation_loss or validation_loss <= best_validation_loss:
					best_validation_loss = validation_loss
					best_model_path = model_path
					best_config = config_params
					best_eval = eval_values

				validation_losses.append(validation_loss)

			elif line.startswith('Perplexity:'):
				perplexity = float(line.split()[-1])
				eval_values["validation_perplexity"] = perplexity
				perplexities.append(perplexity)

		if not eval_values.get('validation_perplexity'):
			eval_values['validation_perplexity'] = -1
			perplexities.append(-1)

		csv += str(eval_values["epoch"])+',' + str(eval_values['validation_loss'])+',' + str(eval_values['validation_perplexity']*args.perplexity_factor)+','

		for split in ['val', 'test']:
			args_sfx = '-'+args.model_iteration+'-'+split+('' if not args.development_mode else '-development_mode')
			args_sfx += '-multinomial' if args.decoding_strategy == 'multinomial' else '-beam_'+str(args.beam_size)+('-gamma_'+str(args.gamma) if args.gamma else '')

			if not csv_header_written:
				csv_header += 'mean utterance length ('+split+'), std dev ('+split+'), bleu-1 ('+split+'), bleu-2 ('+split+'), bleu-3 ('+split+'), bleu-4 ('+split+'), '
				for eval_type in modified_eval_types:
					csv_header += eval_type+'-bleu ('+split+'), '
				for agreement_type in agreement_types:
					csv_header += agreement_type+' agreement ('+split+'), '

			if os.path.exists(config_file.replace('config.txt', 'mul-std-bleu'+args_sfx+'.csv')):
				with open(config_file.replace('config.txt', 'mul-std-bleu'+args_sfx+'.csv')) as f:
					mul_std_bleu = f.readline().strip()
					csv += mul_std_bleu+','
					mul_std_bleu = mul_std_bleu.split(',')

					if 'mean utterance length ('+split+')' in eval_terms:
						eval_values['mean utterance length ('+split+')'] = float(mul_std_bleu[0])

					if 'std dev ('+split+')' in eval_terms:
						eval_values['std dev ('+split+')'] = float(mul_std_bleu[1])

					for i in range(4):
						if 'bleu-'+str(i+1)+' ('+split+')' in eval_terms:
							eval_values['bleu-'+str(i+1)+' ('+split+')'] = float(mul_std_bleu[i+2])

				args_sfx = '-multinomial' if args.decoding_strategy == 'multinomial' else '-beam_'+str(args.beam_size)+('-gamma_'+str(args.gamma) if args.gamma else '')
				generated_sentences_file = config_file.replace('config.txt', 'generated_sentences-best-'+split+args_sfx+'.txt')

				parser = argparse.ArgumentParser()
				parser.add_argument('generated_sentences_file', help='file of sentences generated by a model')
				parser.add_argument('--simple_terms_file', default='../data/lexicons/simple-terms-redux.txt')
				parser.add_argument('--colors_file', default='../data/lexicons/colors.txt')
				parser.add_argument('--spatial_relations_file', default='../data/lexicons/spatial-relations.txt')
				parser.add_argument('--dialogue_file', default='../data/lexicons/dialogue.txt')
				parser.add_argument('--shapes_file', default='../data/lexicons/shapes.txt')
				parser.add_argument('--synonyms_file', default='../data/lexicons/synonym_substitutions.json')
				parser.add_argument('--with_simple_synonyms', default=True, action='store_true')
				parser.add_argument('--with_utterance_synonyms', default=True, action='store_true')
				parser.add_argument('--num_synonym_references', default=4)
				parser.add_argument('--suppress_printing', default=True)
				parser.add_argument('--output_file', default=None)

				modified_bleu_args = parser.parse_args([generated_sentences_file])
				modified_bleu_scores, agreements = evaluation_analysis.main(modified_bleu_args)

				for eval_type in modified_eval_types:
					header_str = eval_type+'-bleu ('+split+')'
					if header_str in eval_terms:
						eval_values[header_str] = modified_bleu_scores[eval_type]
					csv += str(modified_bleu_scores[eval_type])+','

				for agreement_type in agreement_types:
					header_str = agreement_type+' agreement ('+split+')'
					if header_str in eval_terms:
						eval_values[header_str] = agreements[agreement_type]
					csv += str(agreements[agreement_type])+','

			else:
				for i in range(9+len(modified_eval_types)):
					csv += '-1,'

		for param in all_params:
			if config_params.get(param) is None:
				default_value = '-1'
				if all_params[param] == bool:
					default_value = 'False'
				elif all_params[param] == str:
					default_value = 'None'

				config_params[param] = default_value

		pair_values = {'num_encoder_hidden_layers': None, 'num_decoder_hidden_layers': None, 'dropout_rnn': None, 'dropout_nae': None, 'dropout_counter': None}

		for param, value in sorted(config_params.items()):
			if param == 'load_dataset' or param == 'ignore_diff' or param == 'data_dir' or param == 'gold_configs_dir' or param == 'vocab_dir' or param == 'date_dir' or param == 'teacher_forcing_ratio' or param == 'seed' or param == 'num_workers' or param == 'strict' or param == 'suppress_logs' or param == 'visualize':
				continue

			if args.model_type == 'cnn_3d':
				if 'pretrained' in param or param == 'add_builder_utterances' or param == 'augment_dataset' or param == 'augmentation_factor' or param == 'pretrained_and_augmented' or param == 'exactly_k' or param == 'strict' or param == 'ignore_diff' or 'world_state' in param or 'block_embedding' in param:
					continue

			if isinstance(value, str) and ',' in value:
				value = '"'+value+'"'

			csv += str(value)+','
			if not csv_header_written:
				csv_header += param+','

			if param in pair_values and value is not None:
				pair_values[param] = str(value)
			elif param != 'hyperparameter_file':
				append_eval_values(eval_statistics, param, value, eval_values)

		append_eval_values(eval_statistics, 'encoder_decoder_layers', pair_values['num_encoder_hidden_layers']+','+pair_values['num_decoder_hidden_layers'], eval_values)

		if pair_values['dropout_rnn'] is not None and pair_values['dropout_rnn'] != 'None':
			header = 'dropout_rnn_counter'
			pair_value = pair_values['dropout_counter']

			if 'next_actions' in args.model_type:
				header = 'dropout_rnn_nae'
				pair_value = pair_values['dropout_nae']

			append_eval_values(eval_statistics, header, pair_values['dropout_rnn']+','+pair_value, eval_values)

		if not csv_header_written:
			eval_csv_writer.write(csv_header[:-1]+'\n')
			csv_header_written = True

		eval_csv_writer.write(csv[:-1]+'\n')

	if not best_model_path:
		print("Error: no best model was found -- check that the model directories include an eval.txt file.")
		sys.exit(0)

	eval_writer.write('='*89)
	eval_writer.close()
	eval_csv_writer.close()

	# print details of best overall model found
	print("Best model found:", model_path)
	for param in best_config:
		print(param.ljust(25), best_config[param])
	for value in best_eval:
		print(value.ljust(25), best_eval[value])

	# print statistics of min/max/std over all models
	print('\nEpochs at which best models were found:')
	print('\tmin:', np.min(epochs))
	print('\tmax:', np.max(epochs))
	print('\tstd:', np.std(epochs))

	print('\nValidation losses:')
	print('\tmin:', np.min(validation_losses))
	print('\tmax:', np.max(validation_losses))
	print('\tstd:', np.std(validation_losses))

	print('\nPerplexities:')
	print('\tmin:', np.min(perplexities))
	print('\tmax:', np.max(perplexities))
	print('\tstd:', np.std(perplexities))

	print("\nWrote cumulative evaluation log to", os.path.join(args.models_dir, 'cumulative_evals-'+args.model_iteration+'.txt'))
	print("Wrote cumulative evaluation csv to", os.path.join(args.models_dir, 'cumulative_evals-'+args.model_iteration+'.csv'), '\n')

	for param in list(eval_statistics.keys()):
		for value in list(eval_statistics[param].keys()):
			if len(eval_statistics[param][value]) < len(eval_terms):
				eval_statistics[param].pop(value)

		if len(eval_statistics[param]) < 2:
			eval_statistics.pop(param)

	for param in eval_statistics:
		print('Parameter:', param)
		for value in eval_statistics[param]:
			print('\tValue:', value)
			for eval_term in eval_terms:
				eval_values = eval_statistics[param][value][eval_term]
				if len(eval_values) < 1:
					continue

				max_value = "{0:.3f}".format(max(eval_values))
				mean = "{0:.3f}".format(np.mean(eval_values))
				std = "{0:.3f}".format(np.std(eval_values))
				print('\t\t', eval_term.ljust(30), 'max: '+max_value+'  mean: '+mean+'  std: '+std)

		print()

	sys.stdout = sys.__stdout__

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=str, help='path for models saved on a specific date, each containing their own config & eval files')
	parser.add_argument('--perplexity_factor', type=float, default=1.18, help='perplexity factor to multiply against validation perplexity')
	parser.add_argument('--model_type', type=str, default='utterances_and_block_region_counters', help='filter parameters to display based on this model type')
	parser.add_argument('--decoding_strategy', type=str, default='beam', help='multinomial/beam')
	parser.add_argument('--beam_size', type=int, default=10, help='beam size for beam search decoding')
	parser.add_argument('--gamma', type=float, default=0.8, help='gamma penalty')
	parser.add_argument('--model_iteration', default='best', help='iteration of model to be evaluated: "best" or "final"')
	parser.add_argument('--development_mode', default=False, action='store_true', help='accumulate development mode evaluations')
	parser.add_argument('--eval_terms', default='bleu-1 (val), bleu-2 (val), colors-bleu (val), simple_terms-bleu (val)', help='evaluation metrics for which to aggregate mean/std dev statistics over hyperparameter configurations')
	args = parser.parse_args()
	main(args)
