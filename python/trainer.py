import sys, os, argparse, time, itertools, train
from collections import OrderedDict
from utils import timestamp, print_dir, Logger, parse_value
from vocab import Vocabulary
from data_loader import CwCDataset

def flatten_combined_params(model_name, param_lists, combined):
	params = []
	for combined_tuple in combined:
		config = {}
		flattened = flatten(combined_tuple)
		for i in range(len(param_lists)):
			config[list(param_lists.keys())[i]] = flattened[i]

		""" IMPLEMENT ME FOR NEW MODELS """
		if model_name == 'seq2seq':
			if not config.get("linear_size") and config.get("nonlinearity") or config.get("linear_size") and not config.get("nonlinearity"):
				continue

		if model_name == 'cnn_3d':
			if config.get('built_diff_features') != config.get('gold_diff_features'):
				continue

		params.append(config)

	print("Hyperparameter configurations ("+str(len(params))+"):")
	for param in params:
		print("\t", param)
	print()

	return params

def get_param_lists(hyperparameter_file):
	param_lists = OrderedDict()
	with open(hyperparameter_file) as f:
		print(timestamp(), "Reading hyperparameter configuration from", print_dir(hyperparameter_file, 4), "\n")

		for line in f:
			tokens = line.split()
			param = tokens[0]
			values = []

			for value in tokens[1:]:
				values.append(parse_value(value))

			param_lists[param] = values

	print("Parameter lists:", param_lists)
	return param_lists

def combine_params(param_lists):
	combined = None
	for param in param_lists:
		if not combined:
			combined = param_lists[param]
		else:
			combined = itertools.product(combined, param_lists[param])

	return combined

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def main(args):
	""" Training script that runs through different hyperparameter settings and trains multiple models. """
	model_path = os.path.join(args.model_path, args.model)
	timestamp_dir = str(int(round(time.time()*1000)))
	args.date_dir = args.date_dir+'/'+args.model+'_trainer-'+timestamp_dir
	model_path = os.path.join(model_path, args.date_dir)

	if not os.path.exists(model_path) and not args.suppress_logs:
		os.makedirs(model_path)

	log_path = os.path.join(model_path, args.model+'_trainer-'+timestamp_dir+'.log') if not args.suppress_logs else os.devnull
	logger = Logger(log_path)
	sys.stdout = logger

	# create all combinations of hyperparameters
	param_lists = get_param_lists(args.hyperparameter_file)
	combined = combine_params(param_lists)
	params = flatten_combined_params(args.model, param_lists, combined)

	models_trained = 0
	start_time = time.time()

	# train each model
	for i in range(len(params)):
		config = params[i]

		print(timestamp(), "Training model", str(models_trained+1), "of", len(params), "...")
		print(timestamp(), "Parameters tuned:", config)

		for param in config:
			if not hasattr(args, param):
				sys.stdout = sys.__stdout__
				print("Error: you have specified param", param, "but it does not exist as a command-line argument!\nPlease implement this and try again.")
				sys.exit(0)

			setattr(args, param, config[param])

		sys.stdout = sys.__stdout__
		training_log = train.main(args)
		models_trained += 1

		sys.stdout = logger
		print(timestamp(), "Done! Model", str(models_trained), "training log saved to", print_dir(training_log, 6), "\n")

	print(timestamp(), "Model training finished. Number of models trained:", models_trained)
	time_elapsed = time.time()-start_time
	m, s = divmod(time_elapsed, 60)
	h, m = divmod(m, 60)
	print(timestamp(), " Total time elapsed: %d:%02d:%02d (%.2fs)" %(h, m, s, time_elapsed), sep="")
	print(os.path.abspath(model_path))
	sys.stdout = sys.__stdout__

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str, help='type of model to train')
	parser.add_argument('hyperparameter_file', type=str, help='file of hyperparameter options to train models for')

	# io
	parser.add_argument('--model_path', type=str, default='../models/', help='path for saving trained models')
	parser.add_argument('--pretrained_models_dir', type=str, default=None, help='path for pretrained LMs')
	parser.add_argument('--saved_dataset_dir', type=str, default='/shared/data/cwc/scratch/lower-no_perspective_coords-fixed', help='path for saved dataset to use')
	parser.add_argument('--decoder_vocab_path', type=str, default='../vocabulary/glove.42B.300d-lower-2r-speaker-train_split-architect_only/vocab.pkl', help='path for decoder vocabulary wrapper')
	parser.add_argument('--encoder_vocab_path', type=str, default='../vocabulary/glove.42B.300d-lower-1r-speaker-builder_actions-oov_as_unk-all_splits/vocab.pkl', help='path for encoder vocabulary wrapper')
	parser.add_argument('--data_dir', type=str, default='../data/logs/', help='path for data jsons')
	parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/', help='path for gold config xmls')
	parser.add_argument('--vocab_dir', type=str, default="../vocabulary/", help='path for vocabulary files')
	parser.add_argument('--date_dir', type=str, default=time.strftime("%Y%m%d"))

	# dataset options
	parser.add_argument("--load_dataset", default=True, action="store_true", help="Whether to load dataset instead of generating it")
	parser.add_argument("--add_builder_utterances", default=False, action="store_true", help="Whether or not to include builder utterances in the datasets")
	parser.add_argument("--augment_dataset", default=False, action="store_true", help="Whether or not to augment the training dataset -- need to use the right vocab for this to work")
	parser.add_argument('--augmentation_factor', type=int, default=0, help='max #synthetic utterances to be augmented per original utterance')
	parser.add_argument("--exactly_k", default=False, action="store_true", help="Whether to generate exactly k or at most k synthetic utterances per original utterance")
	parser.add_argument("--strict", default=False, action="store_true", help="Whether to be strict about original distribution or not. To be used only when exactly_k is True.")
	parser.add_argument('--num_prev_utterances', type=int, default=1, help='number of previous utterances to use as input')
	parser.add_argument('--blocks_max_weight', type=int, default=1, help='max weight of temporally weighted blocks')
	parser.add_argument('--use_builder_actions', default=False, action='store_true', help='include builder action tokens in the dialogue history')
	parser.add_argument('--feasible_next_placements', default=False, action='store_true', help='whether or not to select from pool of feasible next placements only')
	parser.add_argument('--use_condensed_action_repr', default=False, action='store_true', help='use condensed action representation instead of one-hot')
	parser.add_argument('--action_type_sensitive', default=False, action='store_true', help='use action-type-sensitive representations for blocks')
	parser.add_argument('--spatial_info_window_size', type=int, default=1000, help='window size for region block counters')
	parser.add_argument('--use_existing_blocks_counter', default=False, action='store_true', help='use existing blocks counter in block region counter models')
	parser.add_argument('--counters_extra_feasibility_check', default=False, action='store_true', help='whether or not to make the extra check for conficting blocks')
	parser.add_argument('--ignore_diff', default=False, action='store_true', help='ignore diff when building the dataset')
	parser.add_argument('--augmented_data_fraction', type=float, default=0.0, help='fraction of augmented data to use')

	# training options
	parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs')
	parser.add_argument('--save_per_n_epochs', type=int, default=1, help='save models every n epochs')
	parser.add_argument('--stop_after_n', type=int, default=2, help='stop training models after n epochs of increasing perplexity on the validation set')
	parser.add_argument('--log_step', type=int , default=1000, help='step size for printing log info')
	parser.add_argument('--batch_size', type=int, default=1, help='batch size')
	parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
	parser.add_argument('--seed', type=int, default=1234, help='random seed')
	parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")
	parser.add_argument('--visualize', default=False, action='store_true', help='visualize the model architecture and exit')
	parser.add_argument('--suppress_logs', default=False, action='store_true', help='suppress log messages written to disk')

	# global training hyperparameters
	parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
	parser.add_argument('--decay_lr', default=False, action='store_true', help='whether to decay learning rate')
	parser.add_argument('--encoder_decay_factor', type=float, default=0.1, help='factor by which encoder learning rate will be reduced')
	parser.add_argument('--decoder_decay_factor', type=float, default=0.1, help='factor by which decoder learning rate will be reduced')
	parser.add_argument('--decay_patience', type=int, default=2, help='number of epochs with no improvement after which learning rate will be reduced')
	parser.add_argument('--decoder_learning_ratio', type=float, default=1.0, help='decoder learning ratio')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='teacher forcing ratio')
	parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
	parser.add_argument('--l2_reg', type=float, default=0, help='weight decay')
	parser.add_argument('--dropout', type=float, default=0, help='dropout probability')
	parser.add_argument('--dropout_rnn', type=float, default=None, help='dropout probability of rnn modules only')
	parser.add_argument('--dropout_nae', type=float, default=None, help='dropout probability of next action encoder module only')
	parser.add_argument('--dropout_counter', type=float, default=None, help='dropout probability of counter encoder module only')

	# rnn encoder/decoder hyperparameters
	parser.add_argument('--rnn', type=str, default="gru", help='type of RNN -- gru or lstm')
	parser.add_argument('--rnn_hidden_size', type=int , default=100, help='dimension of lstm hidden states')
	parser.add_argument('--num_encoder_hidden_layers', type=int, default=1, help='number of encoder lstm layers')
	parser.add_argument('--num_decoder_hidden_layers', type=int, default=1, help='number of decoder lstm layers')
	parser.add_argument("--bidirectional", default=False, action="store_true", help="Whether or not to have a bidirectional utterances encoder")
	parser.add_argument('--decoder_linear_size', type=int, default=None, help='size of linear layer after embedding layer in decoder (if desired)')
	parser.add_argument('--decoder_nonlinearity', type=str, default=None, help='type of nonlinearity to use after decoder linear layer (if desired)')
	parser.add_argument('--encoder_linear_size', type=int, default=None, help='size of linear layer after embedding layer in encoder (if desired)')
	parser.add_argument('--encoder_nonlinearity', type=str, default=None, help='type of nonlinearity to use after linear layer in encoder (if desired)')
	parser.add_argument('--attn_model', type=str, default="none", help='type of attention')
	parser.add_argument("--train_embeddings", default=False, action="store_true", help="Whether or not to have trainable embeddings")
	parser.add_argument("--pretrained_decoder", default=False, action="store_true", help="Whether or not to use a pretrained LM decoder")
	parser.add_argument("--pretrained_and_augmented", default=False, action="store_true", help="Whether or not to use a pretrained LM decoder trained w/ data augmentation")

	# world state encoder rnn hyperparameters
	parser.add_argument('--world_state_hidden_size', type=int , default=100, help='dimension of lstm hidden states for world state lstm encoder')
	parser.add_argument('--world_state_num_hidden_layers', type=int , default=1, help='number of world state lstm layers')
	parser.add_argument("--world_state_bidirectional", default=False, action="store_true", help="Whether or not to have a bidirectional world state encoder")

	# block representation hyperparameters
	parser.add_argument('--block_embedding_size', type=int, default=20, help='size of embedding obtained from block input representation')
	parser.add_argument('--block_embedding_layer_nonlinearity', type=str, default="relu", help='type of nonlinearity to use after linear layer for block embeddings')
	parser.add_argument('--use_gold_actions', default=False, action='store_true', help='use gold next action information (oracle), instead of heuristically chosen next actions')
	parser.add_argument('--bypass_block_embedding', default=False, action='store_true', help='bypass embedding the block representation using linear/nonlinear layers')
	parser.add_argument('--pre_concat_block_reprs', default=False, action='store_true', help='concatenate block representations before handing off to embedding layer, as opposed to afterwards')

	parser.add_argument('--counter_embedding_size', type=int, default=15, help='size of embedding obtained from counter input representation')
	parser.add_argument('--counter_embedding_layer_nonlinearity', type=str, default="relu", help='type of nonlinearity to use after linear layer for counter embeddings')
	parser.add_argument('--use_separate_counter_encoders', default=False, action='store_true', help='use separate encoders for counter inputs')
	parser.add_argument('--pre_concat_counter_reprs', default=False, action='store_true', help='concatenate counter representations before handing off to embedding layer, as opposed to afterwards')
	parser.add_argument('--bypass_counter_embedding', default=False, action='store_true', help='bypass embedding the counter representation using linear/nonlinear layers')
	parser.add_argument('--use_global_counters', default=False, action='store_true', help='use global block counters as added input to region block counters encoder')
	parser.add_argument('--use_separate_global_embedding', default=False, action='store_true', help='if using global block counters, use a separate encoder for these inputs')
	parser.add_argument('--global_counter_embedding_size', type=int, default=15, help='if using global block counters and a separate encoder for these inputs, output embedding size of this encoder')

	# 3d cnn hyperparameters
	parser.add_argument('--use_shared_cnn', default=False, action='store_true', help='whether or not to use a shared CNN for built & gold configs in 3D CNN model')
	parser.add_argument('--num_conv_layers', type=int, default=1, help='number of convolutional layers for 3D CNN model')
	parser.add_argument('--num_output_channels', type=int, default=16, help='number of output channels for 3D CNN layers')
	parser.add_argument('--cnn_kernel_size', type=int, default=3, help='size of 3D CNN kernel')
	parser.add_argument('--bn', default=False, action='store_true', help='whether or not to use batch normalization in 3D CNN model')
	parser.add_argument('--maxpool', default=False, action='store_true', help='whether or not to use max pooling in 3D CNN model')
	parser.add_argument('--maxpool_kernel_size', type=int, default=2, help='size of max pool kernel')
	parser.add_argument('--num_fc_layers', type=int, default=1, help='number of fully connected layers for 3D CNN model')
	parser.add_argument('--fc_output_size', type=int, default=400, help='output size of final fully connected layer for 3D CNN model')
	parser.add_argument('--encode_prev_utterances', default=False, action='store_true', help='adds previous utterances RNN encoder to 3D CNN model')
	parser.add_argument('--append_perspective_coords', default=False, action='store_true', help='appends perspective coordinates channels to built config')
	parser.add_argument('--include_empty_channel', default=False, action='store_true', help='includes the empty channel in configuration representations')
	parser.add_argument('--use_diff_type_dists', default=False, action='store_true', help='use type distributions based on configuration diffs')
	parser.add_argument('--built_diff_features', default=None, help='additional features used in the feedforward step of built diff combination')
	parser.add_argument('--gold_diff_features', default=None, help='additional features used in the feedforward step of gold diff combination')
	parser.add_argument('--encode_next_actions', default=False, action='store_true', help='encode predicted next actions')

	# encoder-decoder connection parameters
	parser.add_argument('--set_decoder_hidden', default=False, action='store_true', help='sets decoder hidden state to the decoder_hidden context vector produced by encoder')
	parser.add_argument('--concatenate_decoder_inputs', default=False, action='store_true', help='enables vectors of size decoder_input_concat_size to be concatenated to decoder inputs at every timestep')
	parser.add_argument('--concatenate_decoder_hidden', default=False, action='store_true', help='enables vectors of size decoder_hidden_concat_size to be concatenated to the initial provided decoder hidden state (set_decoder_hidden must be True)')
	parser.add_argument('--decoder_input_concat_size', type=int, default=0, help='size of vector to be concatenated to decoder input at every timestep; if one is not provided by the encoder, a 0-vector of this size is concatenated')
	parser.add_argument('--decoder_hidden_concat_size', type=int, default=0, help='size of vector to be concatenated to decoder hidden state at initialization; if one is not provided by the encoder, a 0-vector of this size is concatenated')
	parser.add_argument('--advance_decoder_t0', default=False, action='store_true', help='advances the decoder at start of sequence by a timestep using the decoder_input_t0 context vector produced by encoder')

	args = parser.parse_args()
	main(args)
