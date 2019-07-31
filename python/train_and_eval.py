import torch, random, sys, torch.nn as nn, re
from utils import *

def train(args, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, train_dl, epoch, visualize=False, params=None):
	"""
		One epoch
	"""
	if encoder:
		encoder.train()
	decoder.train()

	total_loss = 0.0
	total_tokens = 0.0

	total_loss_unmutated = 0.0 # similar to above but not mutated to zero when logging
	total_tokens_unmutated = 0.0 # similar to above but not mutated to zero when logging

	for i, (encoder_inputs, decoder_inputs, decoder_outputs, _) in enumerate(train_dl):
		# one training example
		if args.development_mode and i == 100:
			break

		if encoder_optimizer:
			encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
		encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)

		use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
		if use_teacher_forcing:
			loss = decode(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion, visualize=visualize, params=params)
			if visualize:
				return None

		else: # FIXME: Fix this # NOTE: BROKEN! DO NOT USE!
			print("Error: non-teacher forcing is not supported at this time.")
			sys.exit(0)

			# decoder_input = target_inputs[0].view(1, -1)

			# for t in range(len(target_inputs[0])):
			# 	decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
			# 	_, topi = decoder_output.topk(1) # [64, 1]

			# 	decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
			# 	decoder_input = decoder_input.to(device)
			# 	loss += criterion(decoder_output, target_outputs[0][t])

		target_outputs = decoder_outputs.target_outputs
		normalized_loss = loss / float(len(target_outputs[0]))
		normalized_loss.backward()

		total_tokens += len(target_outputs[0])
		total_tokens_unmutated += len(target_outputs[0])

		if encoder:
			rnn_encoder_modules = list(
				filter(lambda x: isinstance(x, nn.GRU) or isinstance(x, nn.LSTM), list(encoder.modules()))
			)
			if rnn_encoder_modules: # clip only when there is an encoder AND RNNs exists in encoder
				torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip) # TODO: clip gradients for the RNN params only?
		torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

		if encoder_optimizer:
			encoder_optimizer.step()
		decoder_optimizer.step()

		total_loss += loss.item()
		total_loss_unmutated += loss.item()

		# print logging info
		if (args.log_step and i > 0 and i % args.log_step == 0) or (i == len(train_dl) - 1):
			if i == args.log_step:
				batch_size = args.log_step + 1
			elif i == len(train_dl) - 1:
				batch_size = len(train_dl) % args.log_step - 1
			else:
				batch_size = args.log_step
			avg_loss_per_example = total_loss / float(batch_size)
			avg_loss_per_word = total_loss / total_tokens
			print(timestamp(), 'Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Perplexity: %5.4f'
				%(epoch, args.num_epochs, i, len(train_dl) - 1, avg_loss_per_example, np.exp(avg_loss_per_word)))

			total_loss = 0.0
			total_tokens = 0.0

	return EvaluationResult(total_loss_unmutated, total_tokens_unmutated)

def train_mrl(args, encoder, decoder, criterion_xent, criterion_mrl, encoder_optimizer, decoder_optimizer, train_dl, epoch, mrl_factor=1.0, alternate_updates=False, use_xent_criterion=False):
	if encoder:
		encoder.train()
	decoder.train()

	total_loss = 0.0
	total_loss_unmutated = 0.0 # similar to above but not mutated to zero when logging

	for i, (encoder_inputs, decoder_inputs, decoder_outputs, _) in enumerate(train_dl):
		if args.development_mode and i == 100:
			break

		criteria = ['xent', 'mrl']
		if alternate_updates:
			criteria = [criteria[i%2]]

		for ckey in criteria:
			if encoder_optimizer:
				encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
			if use_teacher_forcing:
				if ckey == 'xent':
					encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
					loss = decode(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion_xent)
					target_outputs = decoder_outputs.target_outputs
					normalized_loss = loss / float(len(target_outputs[0]))
					normalized_loss.backward()

				elif ckey == 'mrl':
					encoder_context_1 = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context_1 = initialize_with_context(encoder, decoder, encoder_context_1, args)
					loss1 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context_1, criterion_xent, use_neg_sample=False, use_xent_criterion=use_xent_criterion)
					normalized_loss1 = loss1 / float(len(decoder_outputs.target_outputs[0]))

					encoder_context_2 = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context_2 = initialize_with_context(encoder, decoder, encoder_context_2, args)
					loss2 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context_2, criterion_xent, use_neg_sample=True, use_xent_criterion=use_xent_criterion)
					normalized_loss2 = loss2 / float(len(decoder_outputs.target_outputs_neg[0]))

					target = torch.FloatTensor((1)).fill_(-1)
					target = to_var(target)

					loss = criterion_mrl(normalized_loss1.unsqueeze(0), normalized_loss2.unsqueeze(0), target)
					loss.backward()

			else: # FIXME: Fix this # NOTE: BROKEN! DO NOT USE!
				print("Error: non-teacher forcing is not supported at this time.")
				sys.exit(0)

			if encoder:
				rnn_encoder_modules = list(
					filter(lambda x: isinstance(x, nn.GRU) or isinstance(x, nn.LSTM), list(encoder.modules()))
				)
				if rnn_encoder_modules: # clip only when there is an encoder AND RNNs exists in encoder
					torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip) # TODO: clip gradients for the RNN params only?
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

			if encoder_optimizer:
				encoder_optimizer.step()
			decoder_optimizer.step()

			total_loss += loss.item()
			total_loss_unmutated += loss.item()

		# print logging info
		if (args.log_step and i > 0 and i % args.log_step == 0) or (i == len(train_dl) - 1):
			if i == args.log_step:
				batch_size = args.log_step + 1
			elif i == len(train_dl) - 1:
				batch_size = len(train_dl) % args.log_step - 1
			else:
				batch_size = args.log_step

			avg_loss_per_example = total_loss / float(batch_size)
			print(timestamp(), 'Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f'
				%(epoch, args.num_epochs, i, len(train_dl) - 1, avg_loss_per_example))

			total_loss = 0.0

	return EvaluationResult(total_loss_unmutated)

def eval(args, encoder, decoder, criterion, dev_dl):
	if encoder:
		encoder.eval()
	decoder.eval()

	# compute the validation loss
	validation_loss = 0.0
	num_tokens = 0.0

	with torch.no_grad():
		for i, (encoder_inputs, decoder_inputs, decoder_outputs, _) in enumerate(dev_dl):
			if args.development_mode and i == 1:
				break

			encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
			encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
			loss = decode(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion)
			validation_loss += loss.item()
			target_outputs = decoder_outputs.target_outputs
			num_tokens += len(target_outputs[0])

	return EvaluationResult(validation_loss, num_tokens)

def eval_mrl(args, encoder, decoder, criterion_xent, criterion_mrl, dev_dl, mrl_factor=1.0, use_xent_criterion=False):
	if encoder:
		encoder.eval()
	decoder.eval()

	# compute the validation loss of this epoch's model
	validation_loss = 0.0
	num_tokens = 0.0

	with torch.no_grad():
		for i, (encoder_inputs, decoder_inputs, decoder_outputs, _) in enumerate(dev_dl):
			if args.development_mode and i == 1:
				break

			criteria = ['xent', 'mrl']

			for ckey in criteria:
				if ckey == 'xent':
					encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
					loss = decode(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion_xent)
					validation_loss += loss.item()
					target_outputs = decoder_outputs.target_outputs
					num_tokens += len(target_outputs[0])

				elif ckey == 'mrl':
					encoder_context_1 = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context_1 = initialize_with_context(encoder, decoder, encoder_context_1, args)
					loss1 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context_1, criterion_xent, use_neg_sample=False, use_xent_criterion=use_xent_criterion)
					normalized_loss1 = loss1 / float(len(decoder_outputs.target_outputs[0]))

					encoder_context_2 = encoder(encoder_inputs) if encoder else EncoderContext()
					encoder_context_2 = initialize_with_context(encoder, decoder, encoder_context_2, args)
					loss2 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context_2, criterion_xent, use_neg_sample=True, use_xent_criterion=use_xent_criterion)
					normalized_loss2 = loss2 / float(len(decoder_outputs.target_outputs_neg[0]))

					target = torch.FloatTensor((1)).fill_(-1)
					target = to_var(target)

					loss = criterion_mrl(normalized_loss1.unsqueeze(0), normalized_loss2.unsqueeze(0), target)
					validation_loss += loss.item()
					target_outputs = decoder_outputs.target_outputs
					num_tokens += len(target_outputs[0])

			# encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
			# encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
			# loss1 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion_xent, use_neg_sample=False)
			# normalized_loss1 = loss1 / float(len(decoder_outputs.target_outputs[0]))

			# encoder_context = encoder(encoder_inputs) if encoder else EncoderContext()
			# encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
			# loss2 = decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion_xent, use_neg_sample=True)
			# normalized_loss2 = loss2 / float(len(decoder_outputs.target_outputs_neg[0]))

			# target = torch.FloatTensor((1)).fill_(-1)
			# if torch.cuda.is_available():
			# 	target = target.cuda()
			# target = to_var(target)

			# loss_triplet = mrl_factor*criterion_mrl(normalized_loss1.unsqueeze(0), normalized_loss2.unsqueeze(0), target) + (1-mrl_factor)*((normalized_loss1+normalized_loss2)/2)
			# validation_loss += loss_triplet.item()

	return EvaluationResult(validation_loss, num_tokens)

def decode(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion, visualize=False, params=None):
	target_inputs = to_var(decoder_inputs.target_inputs)
	target_outputs = to_var(decoder_outputs.target_outputs)
	decoder_hidden = encoder_context.decoder_hidden

	loss = 0.0
	for t in range(len(target_inputs[0])):
		# one time step
		decoder_input = target_inputs[0][t].view(1, -1) # Next input is current target
		decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_context)

		if visualize:
			from torchviz import make_dot
			g = make_dot(decoder_output, params)
			g.view()
			break

		loss += criterion(decoder_output, target_outputs[0][t].view(1))

	return loss

def decode_mrl(decoder, decoder_inputs, decoder_outputs, encoder_context, criterion_xent, use_neg_sample=False, use_xent_criterion=False):
	target_inputs = to_var(decoder_inputs.target_inputs) if not use_neg_sample else to_var(decoder_inputs.target_inputs_neg)
	target_outputs = to_var(decoder_outputs.target_outputs) if not use_neg_sample else to_var(decoder_outputs.target_outputs_neg)
	decoder_hidden = encoder_context.decoder_hidden

	loss = 0.0
	for t in range(len(target_inputs[0])):
		decoder_input = target_inputs[0][t].view(1, -1) # Next input is current target
		decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_context)

		if use_xent_criterion:
			loss += criterion_xent(decoder_output, target_outputs[0][t].view(1))

		else:
			m = nn.LogSoftmax()
			decoder_output = m(decoder_output)
			target_output = target_outputs[0][t].item()
			loss += decoder_output.view(-1)[target_output]

	return loss

def initialize_with_context(encoder, decoder, encoder_context, args):
	"""
		Condition decoder on encoder's output appropriately
	"""
	def f(encoder_hidden):
		"""
			use same final encoder hidden state to initialize every decoder layer
			take encoder hidden state from something like [1, 1, 100] -> [2, 1, 100]
		"""
		encoder_hidden_flattened = encoder_hidden.view(-1)
		hidden_size = encoder_hidden_flattened.size()[0]
		zero_vector = torch.zeros(hidden_size)
		if torch.cuda.is_available():
			zero_vector = zero_vector.cuda()
		decoder_hidden_flattened = torch.cat(
			[encoder_hidden_flattened] + [zero_vector] * (decoder.num_hidden_layers - 1)
		)
		decoder_hidden = decoder_hidden_flattened.view(decoder.num_hidden_layers, 1, -1)
		return decoder_hidden

	if not args.concatenate_decoder_hidden:
		encoder_context.decoder_hidden_concat = torch.Tensor([])
		if torch.cuda.is_available():
			encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat.cuda()
	else:
		if not isinstance(encoder_context.decoder_hidden_concat, torch.Tensor) and not encoder_context.decoder_hidden_concat:
			encoder_context.decoder_hidden_concat = f(torch.randn(1, 1, args.decoder_hidden_concat_size))
			if torch.cuda.is_available():
				encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat.cuda()
		elif isinstance(encoder_context.decoder_hidden_concat, tuple):
			encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat[0]

	# set decoder hidden state
	if not args.set_decoder_hidden:
		encoder_context.decoder_hidden = None
	else:
		if not isinstance(encoder_context.decoder_hidden, torch.Tensor) and not encoder_context.decoder_hidden:
			print("ERROR: you specified to initialize decoder hidden state with encoder context, but no context was given.")
			sys.exit(0)

		hidden_concat = encoder_context.decoder_hidden_concat
		encoder_context.decoder_hidden = torch.cat((encoder_context.decoder_hidden, hidden_concat), 2)

		if isinstance(encoder_context.decoder_hidden, tuple): # true in case of lstm
			# TODO: BROKEN -- FIX ME
			encoder_context.decoder_hidden = (f(encoder_context.decoder_hidden[0]), f(encoder_context.decoder_hidden[1]))
		else: # true in case of gru and non-rnn modules
			encoder_context.decoder_hidden = f(encoder_context.decoder_hidden)

	# concatenate context to decoder inputs
	if not args.concatenate_decoder_inputs:
		encoder_context.decoder_input_concat = torch.Tensor([])
		if torch.cuda.is_available():
			encoder_context.decoder_input_concat = encoder_context.decoder_input_concat.cuda()
	else:
		if not isinstance(encoder_context.decoder_input_concat, torch.Tensor) and not encoder_context.decoder_input_concat:
			encoder_context.decoder_input_concat = torch.randn(1, 1, args.decoder_input_concat_size)
			if torch.cuda.is_available():
				encoder_context.decoder_input_concat = encoder_context.decoder_input_concat.cuda()
		elif isinstance(encoder_context.decoder_input_concat, tuple):
			encoder_context.decoder_input_concat = encoder_context.decoder_input_concat[0]

	# advance decoder by one timestep
	if args.advance_decoder_t0:
		decoder_input = encoder_context.decoder_input_t0
		_, encoder_context.decoder_hidden, _ = decoder(decoder_input, encoder_context.decoder_hidden, encoder_context, bypass_embed=True)

	return encoder_context

# BEAM SEARCH DECODING

# FIXME: Avoid use magic strings everywhere for SOS and EOS tokens

class Sentence:
	def __init__(self, decoder_hidden, last_idx, last_idx_sibling_rank, sentence_idxes=[], sentence_scores=[]):
		if(len(sentence_idxes) != len(sentence_scores)):
			raise ValueError("length of indexes and scores should be the same")
		self.decoder_hidden = decoder_hidden
		self.last_idx = last_idx
		self.last_idx_sibling_rank = last_idx_sibling_rank # needed for diverse decoding
		self.sentence_idxes =  sentence_idxes
		self.sentence_scores = sentence_scores

	def likelihoodScore(self):
		if len(self.sentence_scores) == 0:
			return -99999999.999 # TODO: check
		# return mean of sentence_score
		# TODO: Relates to the normalized loss function used when training?
		# NOTE: No need to length normalize when making selection for beam. Only needed during final selection.
		return sum(self.sentence_scores) / len(self.sentence_scores)

	def penalizedLikelihoodScore(self, gamma):
		"""
			Diverse decoding: https://arxiv.org/pdf/1611.08562.pdf
		"""
		return sum(self.sentence_scores) - gamma * self.last_idx_sibling_rank

	def addTopk(self, topi, topv, decoder_hidden, beam_size, voc, EOS_token):
		terminates, sentences = [], []
		for i in range(beam_size):
			if topi[0][i] == EOS_token:
				terminates.append(([voc.idx2word[idx.item()] for idx in self.sentence_idxes] + ['</architect>'], # TODO: need the eos token?
								   self.likelihoodScore())) # tuple(word_list, score_float)
				continue
			idxes = self.sentence_idxes[:] # pass by value
			scores = self.sentence_scores[:] # pass by value
			idxes.append(topi[0][i])
			scores.append(topv[0][i])
			sentences.append(Sentence(decoder_hidden, topi[0][i], i+1, idxes, scores))
		return terminates, sentences # NOTE: terminates can be of size 0 or 1 only

	def toWordScore(self, voc, EOS_token):
		words = []
		for i in range(len(self.sentence_idxes)):
			if self.sentence_idxes[i] == EOS_token: # NOTE: never hit
				words.append('</architect>')
			else:
				words.append(voc.idx2word[self.sentence_idxes[i].item()])
		if self.sentence_idxes[-1] != EOS_token:
			words.append('</architect>')
		return (words, self.likelihoodScore())

def beam_decode(decoder, encoder_context, voc, beam_size, max_length, gamma=None, num_top_sentences=1):
	decoder_hidden = encoder_context.decoder_hidden

	terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
	prev_top_sentences.append(Sentence(decoder_hidden=decoder_hidden, last_idx=voc('<architect>'), last_idx_sibling_rank=1))

	for _ in range(max_length):
		for sentence in prev_top_sentences:
			decoder_input = torch.LongTensor([[sentence.last_idx]]) # NOTE: [1, 1]
			if torch.cuda.is_available():
				decoder_input = decoder_input.cuda()

			decoder_output, decoder_hidden_new, _ = decoder(
				decoder_input, sentence.decoder_hidden, encoder_context
			)

			m = nn.LogSoftmax()
			decoder_output = m(decoder_output) # TODO: check base, dim
			topv, topi = decoder_output.topk(beam_size) # topv : tensor([[-0.4913, -1.9879, -2.4969, -3.6227, -4.0751]])
			term, top = sentence.addTopk(topi, topv, decoder_hidden_new, beam_size, voc, voc('</architect>'))
			terminal_sentences.extend(term)
			next_top_sentences.extend(top)

		if gamma:
			next_top_sentences.sort(key=lambda s: s.penalizedLikelihoodScore(gamma), reverse=True)
		else:
			next_top_sentences.sort(key=lambda s: s.likelihoodScore(), reverse=True)
		prev_top_sentences = next_top_sentences[:beam_size]
		next_top_sentences = []

	terminal_sentences += [sentence.toWordScore(voc, voc('</architect>')) for sentence in prev_top_sentences]
	terminal_sentences.sort(key=lambda x: x[1], reverse=True)

	if num_top_sentences is not None:
		top_terminal_sentences = list(map(lambda x: x[0][:-1], terminal_sentences[:num_top_sentences]))
	else:
		top_terminal_sentences = list(map(lambda x: x[0][:-1], terminal_sentences))

	return top_terminal_sentences # terminal_sentences[0][0][:-1]

def multinomial_decode(decoder, decoder_hidden, encoder_outputs, input_encoding, voc, max_length):
	sentence = [voc('<architect>')]

	for _ in range(max_length):
		decoder_input = torch.LongTensor([[sentence[-1]]]) # NOTE: [1, 1]

		decoder_output, decoder_hidden, _ = decoder(
			decoder_input, decoder_hidden, encoder_outputs, input_encoding
		)
		m = nn.Softmax()
		decoder_output = m(decoder_output)
		sampled_word = torch.multinomial(decoder_output, 1)
		sentence.append(sampled_word.item())
		if sentence[-1] == voc('</architect>'):
			break

	return list(map(lambda x: voc.idx2word[x], sentence[1:-1]))

def generate_for_architect_demo(encoder, decoder, test_dl, decoder_vocab, beam_size, max_length, args, development_mode=False, gamma=None):
	encoder.eval()
	decoder.eval()

	generated_utterances, to_print = [], []
	total_examples = str(len(test_dl)) if not development_mode else '300'

	try:
		with torch.no_grad():
			for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(test_dl):
				if development_mode and i == 300:
					break
				for beam_size_ctr in [beam_size, beam_size+5, beam_size+10]: # try max these 3 beam sizes
					allowed_colors = set([k for k, v in raw_inputs.colors_to_all_actions.items() if v > 0])
					if not allowed_colors:
						generated_utterance = [["great", "!", "we", "are", "done", "!"]]
					else:
						encoder_context = encoder(encoder_inputs)
						# encoder_outputs, encoder_hidden = encoder(encoder_inputs)

						# how to make the connection to the decoder
						encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
						# decoder_hidden, input_encoding_for_decoder = connect_encoder_decoder(init_decoder_with_encoder, encoder_hidden, decoder)

						generated_utterance = beam_decode(decoder, encoder_context, decoder_vocab, beam_size_ctr, max_length, gamma, num_top_sentences=None)

						unique_colors_in_gold = set(list(map(lambda x: x["type"], raw_inputs.gold_config_ss)))
						unique_colors_in_built = set(list(map(lambda x: x["type"], raw_inputs.built_config_ss)))
						spatial_refs = ["left", "right", "top", "bottom", "front", "back", "last", "that"]

						def my_filter(generated_utterance):
							# filter out if utterance is too long -- repetition issues
							if len(generated_utterance) > 20:
								return False

							# filter out if there exists a x b patterns, a by b patterns or unk
							if 'x' in generated_utterance or 'by' in generated_utterance or '<unk>' in generated_utterance:
								return False

							# filter out if there exists at least one token that is not in gold or built config
							for token in generated_utterance:
								if token in type2id and token not in unique_colors_in_gold and token not in unique_colors_in_built:
									return False

							# filter out if there are colors and none of them are allowed (allowed = those colors in diffs)
							colors_in_utterance = set(list(filter(lambda x: x in type2id, generated_utterance)))
							allowed_colors = set([k for k, v in raw_inputs.colors_to_all_actions.items() if v > 0])

							if len(colors_in_utterance) > 0:
								if len(colors_in_utterance & allowed_colors) == 0:
									return False

							# filter out if there exist spatial references when built config is empty
							if not unique_colors_in_built:
								for token in generated_utterance:
									if token in spatial_refs:
										return False

							# filter out if there are spatial relations using blocks not yet placed
							generated_utterance_str = " ".join(generated_utterance)

							all_colors = set(sorted(type2id.keys()))
							colors_not_in_built = all_colors - unique_colors_in_built

							colors_not_in_built_regex = "(" + "|".join(colors_not_in_built) + ")"
							fp_regex_1 = re.compile("on top of the( last)? " + colors_not_in_built_regex)
							fp_regex_2 = re.compile("to the (left|right) of the( last)? " + colors_not_in_built_regex)
							fp_regex_3 = re.compile("in front of the( last)? " + colors_not_in_built_regex)
							if fp_regex_1.search(generated_utterance_str) or fp_regex_2.search(generated_utterance_str) or fp_regex_3.search(generated_utterance_str):
								return False

							return True

						generated_utterance = list(filter(my_filter, generated_utterance))
						generated_utterance = generated_utterance[:1]

						if i % 100 == 0:
							print(timestamp(), '['+str(i)+'/'+total_examples+']', list(map(lambda x: " ".join(x), generated_utterance)))
					if generated_utterance: # if at least one utterance made it through the filters, we are done
						to_print.append(list(map(lambda x: " ".join(x), generated_utterance)))

						generated_utterances.append(
							{
								"prev_utterances": encoder_inputs.prev_utterances,
								"next_actions_raw": raw_inputs.next_actions_raw,
								"gold_next_actions_raw": raw_inputs.gold_next_actions_raw,
								"ground_truth_utterance": decoder_outputs.target_outputs,
								"neg_sample_utterance": decoder_outputs.target_outputs_neg,
								"generated_utterance": generated_utterance,
								"block_counters": encoder_inputs.block_counters
							}
						)

						break
	except KeyboardInterrupt:
		print("Generation ended early; quitting.")

	if not generated_utterances: # no generated utterance made it through the filters across multiple beam sizes
		generated_utterance = [["sorry", ",",  "i", "don't", "understand"]]
		to_print.append(list(map(lambda x: " ".join(x), generated_utterance)))
		generated_utterances.append(
			{
				"prev_utterances": encoder_inputs.prev_utterances,
				"next_actions_raw": raw_inputs.next_actions_raw,
				"gold_next_actions_raw": raw_inputs.gold_next_actions_raw,
				"ground_truth_utterance": decoder_outputs.target_outputs,
				"neg_sample_utterance": decoder_outputs.target_outputs_neg,
				"generated_utterance": generated_utterance,
				"block_counters": encoder_inputs.block_counters
			}
		)

	return generated_utterances, to_print

def generate(encoder, decoder, test_dl, decoder_vocab, beam_size, max_length, args, development_mode=False, gamma=None):
	encoder.eval()
	decoder.eval()

	generated_utterances, to_print = [], []
	total_examples = str(len(test_dl)) if not development_mode else '300'

	try:
		with torch.no_grad():
			for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(test_dl):
				if development_mode and i == 300:
					break

				encoder_context = encoder(encoder_inputs)
				# encoder_outputs, encoder_hidden = encoder(encoder_inputs)

				# how to make the connection to the decoder
				encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)
				# decoder_hidden, input_encoding_for_decoder = connect_encoder_decoder(init_decoder_with_encoder, encoder_hidden, decoder)

				generated_utterance = beam_decode(decoder, encoder_context, decoder_vocab, beam_size, max_length, gamma)

				generated_utterances.append(
					{
						"prev_utterances": encoder_inputs.prev_utterances,
						"next_actions_raw": raw_inputs.next_actions_raw,
						"gold_next_actions_raw": raw_inputs.gold_next_actions_raw,
						"ground_truth_utterance": decoder_outputs.target_outputs,
						"neg_sample_utterance": decoder_outputs.target_outputs_neg,
						"generated_utterance": generated_utterance,
						"block_counters": encoder_inputs.block_counters,
						"json_id": raw_inputs.json_id,
						"sample_id": raw_inputs.sample_id,
						"ground_truth_utterance_raw": raw_inputs.next_utterance_raw
					}
				)

				if i % 100 == 0:
					print(timestamp(), '['+str(i)+'/'+total_examples+']', list(map(lambda x: " ".join(x), generated_utterance)))

				to_print.append(list(map(lambda x: " ".join(x), generated_utterance)))
	except KeyboardInterrupt:
		print("Generation ended early; quitting.")

	return generated_utterances, to_print

def multinomial_generate_seq2seq(encoder, decoder, init_decoder_with_encoder, test_dl, decoder_vocab, beam_size, max_length, development_mode=False):
	encoder.eval()
	decoder.eval()

	generated_utterances = []

	with torch.no_grad():
		for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(test_dl):
			if development_mode and i == 300:
				break

			encoder_outputs, encoder_hidden = encoder(encoder_inputs)

			# how to make the connection to the decoder
			decoder_hidden, input_encoding_for_decoder = connect_encoder_decoder(init_decoder_with_encoder, encoder_hidden, decoder)

			generated_sentences = []
			for i in range(3):
				generated_sentence = multinomial_decode(decoder, decoder_hidden, encoder_outputs, input_encoding_for_decoder, decoder_vocab, max_length)
				generated_sentences.append(generated_sentence)

			generated_utterances.append(
				{
					"prev_utterances": encoder_inputs.prev_utterances,
					"next_actions_raw": raw_inputs.next_actions_raw,
					"ground_truth_utterance": decoder_outputs.target_outputs,
					"generated_utterance": generated_sentences
				}
			)

	return generated_utterances

# FIXME: not changed to use updated connecting encoder to decoder framework -- do not use
def multinomial_generate(decoder, voc, max_length, development_mode=False):
	decoder.eval()

	generated_utterances = []

	with torch.no_grad():
		for i in range(2500):
			if development_mode and i == 300:
				break

			encoder_outputs = None
			encoder_hidden = None
			decoder_hidden = None
			input_encoding_for_decoder = torch.Tensor([])
			if torch.cuda.is_available():
				input_encoding_for_decoder = input_encoding_for_decoder.cuda()

			generated_utterance = multinomial_decode(decoder, decoder_hidden, encoder_outputs, input_encoding_for_decoder, voc, max_length)

			generated_utterances.append({
				"generated_utterance": generated_utterance
			})

	return generated_utterances
