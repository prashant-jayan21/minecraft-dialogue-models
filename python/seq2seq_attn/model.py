import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append('..')
from utils import *

class EncoderRNN(nn.Module):
	"""
	Encoder RNN (GRU/LSTM)
	"""
	# TODO: set up for attention, multiple layers
	# FIXME: use embed_dropout!!
	def __init__(self, vocabulary, hidden_size, num_hidden_layers, dropout=0, linear_size=None, nonlinearity=None, rnn="gru", bidirectional=True, train_embeddings=False):
		super(EncoderRNN, self).__init__()
		# Keep for reference
		self.rnn_hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.dropout = dropout
		self.linear_size = linear_size
		self.bidirectional = bidirectional

		# Define layers
		self.embed = vocabulary.word_embeddings
		if train_embeddings:
			self.embed.weight.requires_grad = True

		self.linear = None
		if linear_size:
			self.linear = nn.Linear(vocabulary.embed_size, self.linear_size)

		self.nonlinearity = None
		if nonlinearity == 'relu':
			self.nonlinearity = nn.ReLU()
		elif nonlinearity == 'tanh':
			self.nonlinearity = nn.Tanh()

		self.final_embedding_size = vocabulary.embed_size if not linear_size else linear_size

		self.embed_dropout = nn.Dropout(p=(dropout if self.linear or self.embed.weight.requires_grad else 0))

		if rnn == "gru":
			self.rnn = nn.GRU(self.final_embedding_size, hidden_size, num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)
		elif rnn == "lstm":
			self.rnn = nn.LSTM(self.final_embedding_size, hidden_size, num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)

		# TODO: Generalize to avoid using magic numbers
		self.input_encoding_size = self.rnn_hidden_size # NOTE: even in bidirectional case because we sum forward and backward final hidden states

		self.init_weights()

	def init_weights(self):
		""" Initializes weights of linear layers with Xavier initialization. """
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.bias.data.zero_()
				nn.init.xavier_uniform_(m.weight)

	def forward(self, encoder_inputs):
		input_seq = to_var(encoder_inputs.prev_utterances)
		input_lengths = encoder_inputs.prev_utterances_lengths
		embedded = self.embed(input_seq)
		embedded = self.embed_dropout(embedded) #[1, 64, 512]  # [1, 1, 300]
		packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
		_, hidden = self.rnn(packed) # hidden: (num_layers * num_directions, batch, hidden_size)

		# take only last layer's hidden state
		hidden = hidden.view(self.num_hidden_layers, 2 if self.bidirectional else 1, 1, self.rnn_hidden_size) # (num_layers, num_directions, batch, hidden_size)
		hidden = hidden[-1] # hidden: (num_directions, batch, hidden_size)

		if self.bidirectional:
			def f(hidden):
				"""
					sum final forward and backward hidden states
					take hidden from something like [2, 1, 100] -> [1, 1, 100]
				"""
				return hidden[0].view(-1) + hidden[1].view(-1)

			if isinstance(self.rnn, nn.GRU):
				hidden = f(hidden)
			elif isinstance(self.rnn, nn.LSTM):
				hidden = ( f(hidden[0]), f(hidden[1]))

		# represent hidden state as one single vector (reshaped though)
		if isinstance(self.rnn, nn.GRU):
			hidden = hidden.view(1, 1, -1)
		elif isinstance(self.rnn, nn.LSTM):
			hidden = (hidden[0].view(1, 1, -1), hidden[1].view(1, 1, -1))

		return EncoderContext(decoder_hidden=hidden, decoder_input_concat=hidden, decoder_hidden_concat=hidden, decoder_input_t0=hidden)

	def flatten_parameters(self):
		self.rnn.flatten_parameters()

class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()

		self.method = method
		self.hidden_size = hidden_size

		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)

		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def forward(self, hidden, encoder_outputs):
		# hidden [1, 64, 512], encoder_outputs [14, 64, 512]
		max_len = encoder_outputs.size(1)
		batch_size = encoder_outputs.size(0)

		# Create variable to store attention energies
		attn_energies = torch.zeros(batch_size, max_len) # B x S

		USE_CUDA = torch.cuda.is_available()
		device = torch.device("cuda" if USE_CUDA else "cpu")
		attn_energies = attn_energies.to(device)

		# For each batch of encoder outputs
		for b in range(batch_size):
			# Calculate energy for each encoder output
			for i in range(max_len):
				attn_energies[b, i] = self.score(hidden[:,b], encoder_outputs[b, i].unsqueeze(0))

		# Normalize energies to weights in range 0 to 1, resize to 1 x B x S
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

	def score(self, hidden, encoder_output):
		# hidden [1, 512], encoder_output [1, 512]
		if self.method == 'dot':
			energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
			return energy

		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = hidden.squeeze(0).dot(energy.squeeze(0))
			return energy

		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden, encoder_output), 1))
			energy = self.v.squeeze(0).dot(energy.squeeze(0))
			return energy

class LuongAttnDecoderRNN(nn.Module):
	"""
	Decoder RNN (GRU/LSTM)
	"""
	def __init__(self, attn_model, vocabulary, hidden_size, num_hidden_layers, dropout=0, input_encoding_size=0, hidden_encoding_size=0, rnn="gru", linear_size=None, nonlinearity=None, train_embeddings=False):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = vocabulary.num_tokens
		self.num_hidden_layers = num_hidden_layers
		self.dropout = dropout
		self.input_encoding_size = input_encoding_size
		self.hidden_encoding_size = hidden_encoding_size
		self.linear_size = linear_size

		# Define layers
		self.embed = vocabulary.word_embeddings
		if train_embeddings:
			self.embed.weight.requires_grad = True

		self.linear = None
		if linear_size:
			self.linear = nn.Linear(vocabulary.embed_size, self.linear_size)

		self.nonlinearity = None
		if nonlinearity == 'relu':
			self.nonlinearity = nn.ReLU()
		elif nonlinearity == 'tanh':
			self.nonlinearity = nn.Tanh()

		self.final_embedding_size = vocabulary.embed_size if not linear_size else linear_size

		self.embed_dropout = nn.Dropout(p=(dropout if self.linear or self.embed.weight.requires_grad else 0))

		self.concat_embedding_size = self.final_embedding_size+self.input_encoding_size
		self.concat_hidden_size = self.hidden_size+self.hidden_encoding_size

		if rnn == "gru":
			self.rnn = nn.GRU(self.concat_embedding_size, self.concat_hidden_size, num_hidden_layers, dropout=dropout, batch_first=True)
		elif rnn == "lstm":
			self.rnn = nn.LSTM(self.concat_embedding_size, self.concat_hidden_size, num_hidden_layers, dropout=dropout, batch_first=True)

		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn(attn_model, hidden_size)
			self.concat = nn.Linear(hidden_size * 2, hidden_size)

		self.out = nn.Linear(self.concat_hidden_size, self.output_size)

		self.init_weights()

	def init_weights(self):
		""" Initializes weights of linear layers with Xavier initialization. """
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.bias.data.zero_()
				nn.init.xavier_uniform_(m.weight)

	def forward(self, input_seq, last_hidden, encoder_context, bypass_embed=False):
		input_encoding = encoder_context.decoder_input_concat
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)

		embedded = input_seq if bypass_embed else self.embed(input_seq)

		# Embed into new space
		if self.linear:
			embedded = self.nonlinearity(self.linear(embedded))

		embedded = self.embed_dropout(embedded) #[1, 64, 512]  # [1, 1, 300]

		# Add input encoding
		if not bypass_embed:
			embedded = torch.cat((embedded, input_encoding), 2)

		if(embedded.size(1) != 1):
			raise ValueError('Decoder input sequence length should be 1')

		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.rnn(embedded, last_hidden)

		if self.attn_model != 'none':
			encoder_outputs = encoder_context.attn_vec
			# Calculate attention from current RNN state and all encoder outputs;
			# apply to encoder outputs to get weighted average
			attn_weights = self.attn(rnn_output, encoder_outputs) #[64, 1, 14]
			# encoder_outputs [14, 64, 512]
			context = attn_weights.bmm(encoder_outputs) #[64, 1, 512]

			# Attentional vector using the RNN hidden state and context vector
			# concatenated together (Luong eq. 5)
			rnn_output = rnn_output.squeeze(0) #[64, 512]
			context = context.squeeze(1) #[64, 512]
			concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
			concat_output = F.tanh(self.concat(concat_input)) #[64, 512]
		else:
			concat_output = rnn_output.squeeze(0)
			attn_weights = None

		# Finally predict next token (Luong eq. 6, without softmax)
		output = self.out(concat_output) #[64, output_size]

		# Return final output, hidden state, and attention weights (for visualization)
		return output, hidden, attn_weights

	def flatten_parameters(self):
		self.rnn.flatten_parameters()

if __name__ == "__main__":
	"""
	Use this section for debugging purposes.
	"""

	import pickle, sys
	sys.path.append('..')
	from vocab import Vocabulary

	with open('../../vocabulary/glove.840B.300d-lower-5r-speaker.pkl', 'rb') as f:
		vocab = pickle.load(f)

	encoder = EncoderRNN(vocab, hidden_size=8, num_hidden_layers=1, dropout=0, linear_size=100, nonlinearity="tanh")
