import sys, os, time, uuid, random, pickle, argparse, collections, csv, torch, torch.nn as nn

sys.path.append('..')
from data_loader import CwCDataset
from utils import *
from vocab import Vocabulary
from train_and_eval import train, eval
from seq2seq_attn.model import LuongAttnDecoderRNN

def main(args):

	""" Trains one model given the specified arguments. """
	start_time = time.time()
	initialize_rngs(args.seed, torch.cuda.is_available())

	# create a (unique) new directory for this model based on timestamp
	model_path = os.path.join(args.model_path, args.model)
	date_dir = args.date_dir
	timestamp_dir = str(int(round(start_time*1000)))
	model_path = os.path.join(model_path, date_dir, timestamp_dir)

	if not args.suppress_logs:
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		else:  # race condition: another model directory at this timestamp already exists, so append a random uuid and try again
			temp_path = model_path
			while os.path.exists(temp_path):
				uuid_rand = str(uuid.uuid4())
				temp_path = model_path+"-"+uuid_rand

			model_path = temp_path
			os.makedirs(model_path)

	log_path = os.path.join(model_path, args.model+'_train.log') if not args.suppress_logs else os.devnull
	sys.stdout = Logger(log_path)

	print(timestamp(), args, '\n')
	print(timestamp(), "Models will be written to", print_dir(model_path, 5))
	print(timestamp(), "Logs will be written to", print_dir(log_path, 6))

	if args.use_builder_actions and 'builder_actions' not in args.encoder_vocab_path:
		print("Error: you specified to use builder action tokens in the dialogue history, but they do not exist in the encoder's vocabulary.")
		sys.exit(0)

	if not args.use_builder_actions and 'builder_actions' in args.encoder_vocab_path:
		print("Warning: you specified not to use builder action tokens, but your encoder vocabulary contained them; resetting vocabulary to default: ../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl")
		args.encoder_vocab_path = '../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl'

	# write the configuration arguments to a config file in the model directory
	if not args.suppress_logs:
		with open(os.path.join(model_path, "config.txt"), "w") as f:
			args_dict = vars(args)
			for param in args_dict:
				f.write(param.ljust(20)+"\t"+str(args_dict[param])+"\n")

	print(timestamp(), "Hyperparameter configuration written to", print_dir(os.path.join(model_path, "config.txt"), 6), "\n")

	# load the vocabularies
	with open(args.decoder_vocab_path, 'rb') as f:
		print(timestamp(), "Loading decoder vocabulary from", print_dir(args.decoder_vocab_path, 3), "...")
		decoder_vocab = pickle.load(f)
		print(timestamp(), "Successfully loaded decoder vocabulary.\n")

	with open(args.encoder_vocab_path, 'rb') as f:
		print(timestamp(), "Loading encoder vocabulary from", print_dir(args.encoder_vocab_path, 3), "...")
		encoder_vocab = pickle.load(f)
		print(timestamp(), "Successfully loaded encoder vocabulary.\n")

	# load train and validation data
	print(timestamp(), "Loading the data ...\n")

	lower_dec = "lower" in os.path.abspath(args.decoder_vocab_path)
	lower_enc = "lower" in os.path.abspath(args.encoder_vocab_path)

	if lower_dec != lower_enc:
		print("Encoder and decoder vocabs have to be cased the same way. Different casing is currently not supported.")
		sys.exit(0)

	if args.load_dataset and (lower_dec and "lower" not in os.path.abspath(args.saved_dataset_dir)) or (not lower_dec and "lower" in os.path.abspath(args.saved_dataset_dir)):
		print("Vocabulary and dataset should be cased the same way.")
		sys.exit(0)

	lower = lower_dec

	if args.augment_dataset and args.model != "lm":
		print("Error: Trying to augment training dataset for a model other than a language model.")
		sys.exit(0)

	train_dataset = CwCDataset(
		model=args.model, split="train", lower=lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff,
		augment_dataset=args.augment_dataset, augmentation_factor=args.augmentation_factor, exactly_k=args.exactly_k, strict=args.strict,
		data_dir=args.data_dir, gold_configs_dir=args.gold_configs_dir, saved_dataset_dir=args.saved_dataset_dir, vocab_dir=args.vocab_dir,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, load_dataset=args.load_dataset, transform=None, augmented_data_fraction=args.augmented_data_fraction
	)

	train_dataset.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, include_empty_channel=args.include_empty_channel, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)
	train_dl = train_dataset.get_data_loader(batch_size=1, shuffle=True, num_workers=args.num_workers)

	dev_dataset = CwCDataset(
		model=args.model, split="val", lower=lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff,
		augment_dataset=args.augment_dataset, augmentation_factor=args.augmentation_factor, exactly_k=args.exactly_k, strict=args.strict,
		data_dir=args.data_dir, gold_configs_dir=args.gold_configs_dir, saved_dataset_dir=args.saved_dataset_dir, vocab_dir=args.vocab_dir,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, load_dataset = args.load_dataset, transform=None
	)

	dev_dataset.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, include_empty_channel=args.include_empty_channel, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)
	dev_dl = dev_dataset.get_data_loader(batch_size=1, shuffle=True, num_workers=args.num_workers)

	print(timestamp(), "Successfully loaded the data.\n")

	# initialize the model
	print(timestamp(), "Initializing the model ...\n")

	""" IMPLEMENT ME FOR NEW MODELS """
	if args.model == 'seq2seq_attn':
		from seq2seq_attn.model import EncoderRNN
		encoder = EncoderRNN(
			encoder_vocab, args.rnn_hidden_size, args.num_encoder_hidden_layers, dropout=args.dropout, linear_size=args.encoder_linear_size, nonlinearity=args.encoder_nonlinearity, rnn=args.rnn, bidirectional=args.bidirectional, train_embeddings=args.train_embeddings
		)
	elif args.model == 'seq2seq_world_state':
		from seq2seq_world_state.model import WorldStateEncoderRNN
		encoder = WorldStateEncoderRNN(
			block_input_size=train_dl.dataset.src_input_size_configs, block_embedding_size=args.block_embedding_size, block_embedding_layer_nonlinearity=args.block_embedding_layer_nonlinearity,
			hidden_size=args.world_state_hidden_size, num_hidden_layers=args.world_state_num_hidden_layers,
			bidirectional=args.world_state_bidirectional, dropout=args.dropout, rnn=args.rnn #TODO: Check block_input_size # TODO: Separate RNNs?
		)
	elif args.model == 'world_state_next_actions':
		from seq2seq_world_state.model import NextActionsEncoder
		encoder = NextActionsEncoder(
			block_input_size=train_dl.dataset.src_input_size_next_actions, block_embedding_size=args.block_embedding_size, block_embedding_layer_nonlinearity=args.block_embedding_layer_nonlinearity,
			dropout=args.dropout #TODO: Check block_input_size
		)
	elif args.model == 'utterances_and_next_actions':
		from seq2seq_all_inputs.model import UtterancesAndNextActionsEncoder
		encoder = UtterancesAndNextActionsEncoder(args, train_dl, encoder_vocab)
	elif args.model == 'utterances_and_block_counters':
		from seq2seq_all_inputs.model import UtterancesAndBlockCountersEncoder
		encoder = UtterancesAndBlockCountersEncoder(args, train_dl, encoder_vocab)
	elif args.model == 'utterances_and_block_region_counters':
		from seq2seq_all_inputs.model import UtterancesAndBlockRegionCountersEncoder
		encoder = UtterancesAndBlockRegionCountersEncoder(args, train_dl, encoder_vocab)
	elif args.model == 'seq2seq_all_inputs':
		from seq2seq_all_inputs.model import AllInputsEncoder
		encoder = AllInputsEncoder(args, train_dl, encoder_vocab)
	elif args.model == 'cnn_3d':
		from cnn_3d.model import WorldStateEncoderCNN

		cnn_output_size = args.rnn_hidden_size

		if args.advance_decoder_t0:
			cnn_output_size = decoder_vocab.embed_size+args.decoder_input_concat_size
		elif args.concatenate_decoder_inputs:
			cnn_output_size = args.decoder_input_concat_size
			if args.encode_next_actions:
				cnn_output_size -= args.block_embedding_size*2
		elif args.concatenate_decoder_hidden:
			cnn_output_size = args.decoder_hidden_concat_size

		encoder = WorldStateEncoderCNN(
			cnn_output_size=cnn_output_size, args=args, train_dl=train_dl, encoder_vocab=encoder_vocab
		)
		print()
	elif args.model == 'lm':
		encoder = None
	else:
		print("Error: you have specified model", args.model, "but did not instantiate the appropriate Torch module for the model.\nPlease implement this and try again.")
		sys.exit(0)

	if encoder and not args.set_decoder_hidden and not args.concatenate_decoder_inputs and not args.advance_decoder_t0:
		print("Error: your model contains an encoder module, but you have not specified how its outputs should be connected to the decoder.\nPlease set --set_decoder_hidden, --concatenate_decoder_inputs with --decoder_input_concat_size, or --advance_decoder_t0 and try again.")
		sys.exit(0)

	if args.concatenate_decoder_inputs and args.decoder_input_concat_size == 0:
		print("Error: you specified concatenated inputs (--concatenate_decoder_inputs) for the decoder, but did not specify a size (--decoder_input_concat_size).\nPlease set this appropriately and try again.")
		sys.exit(0)

	if args.concatenate_decoder_hidden and args.decoder_hidden_concat_size == 0:
		print("Error: you specified concatenated hiddens (--concatenate_decoder_hidden) for the decoder, but did not specify a size (--decoder_hidden_concat_size).\nPlease set this appropriately and try again.")
		sys.exit(0)

	if args.pretrained_decoder:
		model_info = []
		with open(os.path.join(args.pretrained_models_dir, "cumulative_evals.csv")) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				row["hidden_size"] = int(row["hidden_size"])
				row["num_decoder_hidden_layers"] = int(row["num_decoder_hidden_layers"])
				row[" validation perplexity"] = float(row[" validation perplexity"])
				row["linear_size"] = int(row["linear_size"])
				row["augment_dataset"] = row["augment_dataset"] == "True"
				row["augmentation_factor"] = int(row["augmentation_factor"])
				row["exactly_k"] = row["exactly_k"] == "True" if "exactly_k" in row else False
				row["strict"] = row["strict"] == "True" if "strict" in row else False
				row["dummy_input_encoding_size"] = int(row["dummy_input_encoding_size"]) if row.get("dummy_input_encoding_size") else 0 # extra check needed for legacy LMs that don't have this dict key
				model_info.append(row)

		model_info = sorted(model_info, key = lambda x: x[" validation perplexity"])

		def g(x):
			return x["decoder_vocab_path"] == args.decoder_vocab_path and x["hidden_size"] == args.rnn_hidden_size and \
			x["num_decoder_hidden_layers"] == args.num_decoder_hidden_layers and x["rnn"] == args.rnn and \
			x["linear_size"] == args.decoder_linear_size and x["nonlinearity"] == args.decoder_nonlinearity and \
			x["augment_dataset"] == args.pretrained_and_augmented and x["augmentation_factor"] == args.augmentation_factor and \
			x["exactly_k"] == args.exactly_k and x["strict"] == args.strict and \
			x["dummy_input_encoding_size"] == input_encoding_size

		try:
			desired_model_info = next(x for x in model_info if g(x))
		except StopIteration:
			print("No matching pretrained decoder found. Exiting...")
			return log_path
		desired_model_path = os.path.join(desired_model_info["saved model path"], "lm-decoder-best.pkl")

		print("Loading pretrained decoder from", desired_model_path)
		with open(desired_model_path, 'rb') as f:
			if not torch.cuda.is_available():
				decoder = torch.load(f, map_location="cpu")
			else:
				decoder = torch.load(f)
		print("Done!")
		# TODO: freeze embeddings if they were unfrozen, check with args.train_embeddings

	else:
		decoder = LuongAttnDecoderRNN(
			args.attn_model, decoder_vocab, args.rnn_hidden_size, args.num_decoder_hidden_layers,
			dropout=args.dropout_rnn if args.dropout_rnn is not None else args.dropout, input_encoding_size=args.decoder_input_concat_size, hidden_encoding_size=args.decoder_hidden_concat_size,
			rnn=args.rnn, linear_size=args.decoder_linear_size, nonlinearity=args.decoder_nonlinearity, train_embeddings=args.train_embeddings
		)

	if encoder:
		print(encoder, '\n')
	print(decoder, '\n')

	# cuda
	if torch.cuda.is_available():
		if encoder:
			encoder.cuda()
		decoder.cuda()

	# initialize optimizer and set loss function
	encoder_optimizer = None
	if encoder:
		encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
		encoder_optimizer = torch.optim.Adam(encoder_parameters, lr=args.learning_rate, weight_decay=args.l2_reg)

	decoder_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
	decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=args.learning_rate * args.decoder_learning_ratio, weight_decay=args.l2_reg)

	encoder_scheduler, decoder_scheduler = None, None
	if args.decay_lr:
		if encoder:
			encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=args.encoder_decay_factor, patience=args.decay_patience, verbose=True)
		decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=args.decoder_decay_factor, patience=args.decay_patience, verbose=True)

	if encoder:
		print("Encoder parameters:")
		for name, param in encoder.named_parameters():
			if param.requires_grad:
				print("  ", name.ljust(30), param.data.size())
		print()

	print("Decoder parameters:")
	for name, param in decoder.named_parameters():
		if param.requires_grad:
			print("  ", name.ljust(30), param.data.size())
	print()

	criterion = nn.CrossEntropyLoss(size_average=False)

	best_epoch, best_eval_result, best_validation_loss = None, None, None
	final_epoch, final_eval_result, final_validation_loss = None, None, None

	increasing = 0	 # number of epochs for which validation loss has steadily increased wrt the global minimum

	print(timestamp(), 'Training the model for a maximum of', args.num_epochs, 'epochs.')
	if args.stop_after_n > 0:
		print(timestamp(), 'Model training will be stopped early if validation loss increases wrt the best validation loss continuously for', args.stop_after_n, 'epochs.')

	print('\n'+timestamp(), "Training the model ...\n")

	try:
		# per epoch
		for epoch in range(1, args.num_epochs+1):
			epoch_start_time = time.time()

			train_result = train(args, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, train_dl, epoch, visualize=args.visualize, params=dict(list(encoder.named_parameters())+list(decoder.named_parameters())))

			if args.visualize:
				break

			eval_result = eval(args, encoder, decoder, criterion, dev_dl)
			validation_loss = eval_result("Loss")

			print('-'*89)
			print(timestamp(), 'End of epoch %d | Time elapsed: %5.2fs' %(epoch, time.time()-epoch_start_time))
			print(timestamp(), 'Training stats |', train_result)
			print(timestamp(), 'Validation stats |', eval_result)

			# save the model per n epochs
			if args.save_per_n_epochs > 0 and epoch % args.save_per_n_epochs == 0:
				print(timestamp(), 'Saving model at epoch %d to %s ...' %(epoch, print_dir(os.path.join(model_path, args.model+'-(encoder/decoder)-epoch-%d.pkl' %(epoch)), 6)))

				if not args.suppress_logs:
					if encoder:
						torch.save(encoder, os.path.join(model_path, args.model+'-encoder-epoch-%d.pkl' %(epoch)))
					torch.save(decoder, os.path.join(model_path, args.model+'-decoder-epoch-%d.pkl' %(epoch)))

			# record if this validation loss was best seen so far over epochs
			if not best_validation_loss or validation_loss <= best_validation_loss:
				print(timestamp(), 'Best model so far found at epoch %d.' %(epoch))

				if not args.suppress_logs:
					if encoder:
						torch.save(encoder, os.path.join(model_path, args.model+'-encoder-best.pkl'))
					torch.save(decoder, os.path.join(model_path, args.model+'-decoder-best.pkl'))

				best_validation_loss = validation_loss
				best_eval_result = eval_result
				best_epoch = epoch
				increasing = 0
			else:
				increasing += 1

			if not args.suppress_logs:
				if encoder:
					torch.save(encoder, os.path.join(model_path, args.model+'-encoder-final.pkl'))
				torch.save(decoder, os.path.join(model_path, args.model+'-decoder-final.pkl'))

			final_epoch, final_eval_result, final_validation_loss = epoch, eval_result, validation_loss

			print(timestamp(), 'Validation loss has increased wrt the best for the last', str(increasing), 'epoch(s).')

			# stop early if validation loss has steadly increased for too many epochs
			if args.stop_after_n > 0 and increasing >= args.stop_after_n:
				print(timestamp(), 'Validation loss has increased wrt the best for the last', str(args.stop_after_n), 'epochs; quitting early.')
				raise KeyboardInterrupt

			if encoder_scheduler:
				encoder_scheduler.step(validation_loss)

			if decoder_scheduler:
				decoder_scheduler.step(validation_loss)

			print('-'*89)

	except KeyboardInterrupt:  # exit gracefully if ctrl-C is used to stop training early
		print('-'*89)
		print(timestamp(), 'Exiting from training early...')
		time.sleep(0.1)

	print(timestamp(), 'Done!')

	# print stats about best overall model found and save model accordingly
	if best_validation_loss:
		print(timestamp(), ' Best model was found at epoch %d' %(best_epoch), ' ('+best_eval_result.pretty_print()+').', sep='')

		# write evaluation stats to eval file in model directory
		if not args.suppress_logs:
			with open(os.path.join(model_path, "eval-best.txt"), "w") as f:
				f.write("Best model found at epoch %d.\n" %(best_epoch))
				f.write(best_eval_result.pretty_print('\n'))

	if final_validation_loss:
		print(timestamp(), ' Final model at end of training epoch %d' %(final_epoch), ' ('+final_eval_result.pretty_print()+').', sep='')

		# write evaluation stats to eval file in model directory
		if not args.suppress_logs:
			with open(os.path.join(model_path, "eval-final.txt"), "w") as f:
				f.write("Final model found at epoch %d.\n" %(final_epoch))
				f.write(final_eval_result.pretty_print('\n'))

	print(timestamp(), "Wrote log to:", print_dir(log_path, 6))

	# compute overall time elapsed
	time_elapsed = time.time()-start_time
	m, s = divmod(time_elapsed, 60)
	h, m = divmod(m, 60)
	print(timestamp(), " Total time elapsed: %d:%02d:%02d (%.2fs)" %(h, m, s, time_elapsed), sep="")
	print("="*89,"\n")

	sys.stdout = sys.__stdout__

	return log_path

if __name__ == '__main__':
	# TODO: add args for sample_filters in CwCDataset -- when use case arises
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str, nargs='?', default='seq2seq_attn', help='type of model to train')

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
	parser.add_argument('--block_embedding_size', type=int, default=39, help='size of embedding obtained from block input representation')
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
