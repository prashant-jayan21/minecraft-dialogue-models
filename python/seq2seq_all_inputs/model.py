import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append('..')
from utils import *
from seq2seq_attn.model import EncoderRNN as PrevUtterancesEncoder
from seq2seq_world_state.model import NextActionsEncoder, BlockCountersEncoder, BlockRegionCountersEncoder

class AllInputsEncoder(nn.Module):

	# NOTE: no need of init weights here as that is done within the sub-modules of this module

    def __init__(self, args, train_dl, vocab):
        super(AllInputsEncoder, self).__init__()

        self.world_state_encoder = WorldStateEncoderRNN(
            block_input_size = train_dl.dataset.src_input_size, block_embedding_size = args.block_embedding_size,
            block_embedding_layer_nonlinearity = args.block_embedding_layer_nonlinearity,
            hidden_size = args.world_state_hidden_size,
            num_hidden_layers = args.world_state_num_hidden_layers, dropout = args.dropout, rnn = args.rnn
        )
        self.prev_utterances_encoder = PrevUtterancesEncoderRNN(
            vocabulary = vocab, hidden_size = args.hidden_size,
            num_hidden_layers = args.num_encoder_hidden_layers,
            dropout = args.dropout, linear_size = args.linear_size,
            nonlinearity = args.nonlinearity, rnn = args.rnn,
            bidirectional=args.bidirectional, unfreeze_embeddings=args.unfreeze_embeddings
        )

        self.rnn = args.rnn

        # TODO: Generalize to avoid using magic numbers
        num_directions = 2 if self.prev_utterances_encoder.bidirectional else 1
        self.input_encoding_size = self.world_state_encoder.hidden_size * 2 + self.prev_utterances_encoder.hidden_size * num_directions

    def forward(self, encoder_inputs):
        world_state_hidden_final = self.world_state_encoder(encoder_inputs)
        prev_utterances_outputs, prev_utterances_hidden_final = self.prev_utterances_encoder(encoder_inputs)

        # concatenate both hidden states
        if self.rnn == "gru":
            input_encoding = torch.cat((world_state_hidden_final, prev_utterances_hidden_final), 2)
        elif self.rnn == "lstm":
            input_encoding = (
                torch.cat((world_state_hidden_final[0], prev_utterances_hidden_final[0]), 2),
                torch.cat((world_state_hidden_final[1], prev_utterances_hidden_final[1]), 2)
            )

        return prev_utterances_outputs, input_encoding

class UtterancesAndNextActionsEncoder(nn.Module):

    def __init__(self, args, train_dl, encoder_vocab):
        super(UtterancesAndNextActionsEncoder, self).__init__()

        self.next_actions_encoder = NextActionsEncoder(
			block_input_size=train_dl.dataset.src_input_size_next_actions, block_embedding_size=args.block_embedding_size, block_embedding_layer_nonlinearity=args.block_embedding_layer_nonlinearity,
			dropout=args.dropout_nae if args.dropout_nae is not None else args.dropout, use_gold_actions=args.use_gold_actions, bypass_embed=args.bypass_block_embedding, pre_concat=args.pre_concat_block_reprs
		)

        self.prev_utterances_encoder = PrevUtterancesEncoder(
			encoder_vocab, args.rnn_hidden_size, args.num_encoder_hidden_layers, dropout=args.dropout_rnn if args.dropout_rnn is not None else args.dropout, linear_size=args.encoder_linear_size, nonlinearity=args.encoder_nonlinearity, rnn=args.rnn, bidirectional=args.bidirectional, train_embeddings=args.train_embeddings
		)

        self.input_encoding_size = self.next_actions_encoder.input_encoding_size

    def forward(self, encoder_inputs):
        next_actions_encoding = self.next_actions_encoder(encoder_inputs).decoder_input_concat
        rnn_hidden = self.prev_utterances_encoder(encoder_inputs).decoder_hidden

        return EncoderContext(decoder_hidden=rnn_hidden, decoder_input_concat=next_actions_encoding)

class UtterancesAndBlockCountersEncoder(nn.Module):
    """
        Integrated model -- combines an encoder RNN for encoding previous utterances with a global block counters encoder
    """
    def __init__(self, args, train_dl, encoder_vocab):
        super(UtterancesAndBlockCountersEncoder, self).__init__()

        self.block_counters_encoder = BlockCountersEncoder(
            input_size=6, output_embedding_size=args.counter_embedding_size, embedding_layer_nonlinearity=args.counter_embedding_layer_nonlinearity,
            dropout=args.dropout_counter if args.dropout_counter is not None else args.dropout, use_separate_encoders=args.use_separate_counter_encoders, pre_concat=args.pre_concat_counter_reprs, bypass_embed=args.bypass_counter_embedding
        )

        self.prev_utterances_encoder = PrevUtterancesEncoder(
            encoder_vocab, args.rnn_hidden_size, args.num_encoder_hidden_layers, dropout=args.dropout_rnn if args.dropout_rnn is not None else args.dropout, linear_size=args.encoder_linear_size, nonlinearity=args.encoder_nonlinearity, rnn=args.rnn, bidirectional=args.bidirectional, train_embeddings=args.train_embeddings
        )

        self.input_encoding_size = self.block_counters_encoder.input_encoding_size

    def forward(self, encoder_inputs):
        block_counters_encoding = self.block_counters_encoder(encoder_inputs).decoder_input_concat
        rnn_hidden = self.prev_utterances_encoder(encoder_inputs).decoder_hidden

        return EncoderContext(decoder_hidden=rnn_hidden, decoder_input_concat=block_counters_encoding, decoder_hidden_concat=block_counters_encoding)

    def flatten_parameters(self):
        self.prev_utterances_encoder.flatten_parameters()

class UtterancesAndBlockRegionCountersEncoder(nn.Module):
    """
        Integrated model -- combines an encoder RNN for encoding previous utterances with a regional block counters encoder (which comes with an optional global block counters encoder as well)
    """
    def __init__(self, args, train_dl, encoder_vocab):
        super(UtterancesAndBlockRegionCountersEncoder, self).__init__()

        input_size_per_region = 24 if args.use_existing_blocks_counter else 18

        self.block_region_counters_encoder = BlockRegionCountersEncoder(
            input_size=input_size_per_region*(33 if args.spatial_info_window_size > 1 else 27)+1, output_embedding_size=args.counter_embedding_size, embedding_layer_nonlinearity=args.counter_embedding_layer_nonlinearity,
            dropout=args.dropout_counter if args.dropout_counter is not None else args.dropout, use_separate_encoders=args.use_separate_counter_encoders, pre_concat=args.pre_concat_counter_reprs, bypass_embed=args.bypass_counter_embedding,
            use_global_counters=args.use_global_counters, use_separate_global_embedding=args.use_separate_global_embedding, global_counter_embedding_size=args.global_counter_embedding_size,
            use_existing_blocks_counter=args.use_existing_blocks_counter
        )

        self.prev_utterances_encoder = PrevUtterancesEncoder(
            encoder_vocab, args.rnn_hidden_size, args.num_encoder_hidden_layers, dropout=args.dropout_rnn if args.dropout_rnn is not None else args.dropout, linear_size=args.encoder_linear_size, nonlinearity=args.encoder_nonlinearity, rnn=args.rnn, bidirectional=args.bidirectional, train_embeddings=args.train_embeddings
        )

        self.input_encoding_size = self.block_region_counters_encoder.input_encoding_size

    def forward(self, encoder_inputs):
        block_counters_encoding = self.block_region_counters_encoder(encoder_inputs).decoder_input_concat
        rnn_hidden = self.prev_utterances_encoder(encoder_inputs).decoder_hidden

        return EncoderContext(decoder_hidden=rnn_hidden, decoder_input_concat=block_counters_encoding, decoder_hidden_concat=block_counters_encoding)

    def flatten_parameters(self):
        self.prev_utterances_encoder.flatten_parameters()
