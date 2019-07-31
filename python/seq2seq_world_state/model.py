import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append('..')
from utils import *

class WorldStateEncoderRNN(nn.Module):
    # TODO: set up for attention, multiple layers
    def __init__(self, block_input_size, block_embedding_size, block_embedding_layer_nonlinearity, hidden_size, num_hidden_layers, bidirectional=False, dropout=0, rnn="lstm"):
        super(WorldStateEncoderRNN, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Define layers
        self.block_embedding_layer = nn.Linear(block_input_size, block_embedding_size)

        if block_embedding_layer_nonlinearity == 'relu':
            self.block_embedding_layer_nonlinearity = nn.ReLU()
        elif block_embedding_layer_nonlinearity == 'tanh':
            self.block_embedding_layer_nonlinearity = nn.Tanh()

        self.final_embedding_size = block_embedding_size

        self.embed_dropout = nn.Dropout(p=dropout)

        if rnn == "gru":
            self.rnn = nn.GRU(self.final_embedding_size, self.hidden_size, self.num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)
        elif rnn == "lstm":
            self.rnn = nn.LSTM(self.final_embedding_size, self.hidden_size, self.num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)

        self.input_encoding_size = self.hidden_size # NOTE: even in bidirectional case because we sum forward and backward final hidden states

        self.init_weights()

    def init_weights(self):
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def forward(self, encoder_inputs):
        built_config = to_var(encoder_inputs.built_configs)
        gold_config = to_var(encoder_inputs.gold_configs)
        built_config_length = encoder_inputs.built_config_lengths
        gold_config_length = encoder_inputs.gold_config_lengths

        built_config_embedding = self.block_embedding_layer_nonlinearity(self.block_embedding_layer(built_config))
        built_config_embedding = self.embed_dropout(built_config_embedding)
        packed_built_config_embedding = pack_padded_sequence(built_config_embedding, built_config_length, batch_first=True)

        _, hidden_built = self.rnn(packed_built_config_embedding) # [1, 1, 100]

        gold_config_embedding = self.block_embedding_layer_nonlinearity(self.block_embedding_layer(gold_config))
        gold_config_embedding = self.embed_dropout(gold_config_embedding)
        packed_gold_config_embedding = pack_padded_sequence(gold_config_embedding, gold_config_length, batch_first=True)

        _, hidden_gold = self.rnn(packed_gold_config_embedding) # [1, 1, 100]

        if self.bidirectional:
            def f(hidden):
                """
                    sum final forward and backward hidden states
                    take hidden from something like [2, 1, 100] -> [1, 1, 100]
                """
                return (hidden[0].view(-1) + hidden[1].view(-1)).view(1, 1, -1)

            if isinstance(self.rnn, nn.GRU):
                hidden_built = f(hidden_built)
                hidden_gold = f(hidden_gold)
            elif isinstance(self.rnn, nn.LSTM):
                hidden_built = ( f(hidden_built[0]), f(hidden_built[1]) )
                hidden_gold = ( f(hidden_gold[0]), f(hidden_gold[1]) )

        if isinstance(self.rnn, nn.GRU):
            hidden = hidden_built + hidden_gold
        elif isinstance(self.rnn, nn.LSTM):
            hidden = ( hidden_built[0] + hidden_gold[0], hidden_built[1] + hidden_gold[1] )

        return None, hidden # TODO: Replace None w/ real outputs when needed

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

class NextActionsEncoder(nn.Module):
    def __init__(self, block_input_size, block_embedding_size, block_embedding_layer_nonlinearity, dropout=0, use_gold_actions=False, bypass_embed=False, pre_concat=False):
        super(NextActionsEncoder, self).__init__()
        # Keep for reference
        self.dropout = dropout
        self.use_gold_actions = use_gold_actions
        self.bypass_embed = bypass_embed
        self.pre_concat = pre_concat

        # Define layers
        if not self.bypass_embed:
            self.block_embedding_layer = nn.Linear(block_input_size if not self.pre_concat else block_input_size*2, block_embedding_size)

            if block_embedding_layer_nonlinearity == 'relu':
                self.block_embedding_layer_nonlinearity = nn.ReLU()
            elif block_embedding_layer_nonlinearity == 'tanh':
                self.block_embedding_layer_nonlinearity = nn.Tanh()

            self.embed_dropout = nn.Dropout(p=dropout)

        self.input_encoding_size = block_embedding_size if self.bypass_embed or self.pre_concat else block_embedding_size * 2 # FIXME: remove magic number

        self.init_weights()

    def init_weights(self):
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def forward(self, encoder_inputs):
        next_placements = to_var(encoder_inputs.next_actions["next_placements"]) if not self.use_gold_actions else to_var(encoder_inputs.gold_actions["gold_placements"])
        next_removals = to_var(encoder_inputs.next_actions["next_removals"]) if not self.use_gold_actions else to_var(encoder_inputs.gold_actions["gold_removals"])

        if self.bypass_embed or self.pre_concat:
            final_encoding = torch.cat((next_placements, next_removals), dim=2)

            if not self.bypass_embed:
                final_encoding = self.block_embedding_layer_nonlinearity(self.block_embedding_layer(final_encoding))
                final_encoding = self.embed_dropout(final_encoding)

        else:
            next_placements_embedding = self.block_embedding_layer_nonlinearity(self.block_embedding_layer(next_placements))
            next_placements_embedding = self.embed_dropout(next_placements_embedding) # [1, 1, embedding size]

            next_removals_embedding = self.block_embedding_layer_nonlinearity(self.block_embedding_layer(next_removals))
            next_removals_embedding = self.embed_dropout(next_removals_embedding) # [1, 1, embedding size]

            final_encoding = torch.cat((next_placements_embedding, next_removals_embedding), dim=2) # [1, 1, 2 * embedding size]

        return EncoderContext(decoder_hidden=final_encoding, decoder_input_concat=final_encoding, decoder_hidden_concat=final_encoding, decoder_input_t0=final_encoding)

class BlockCountersEncoder(nn.Module):
    """
        Global block counters encoder
    """
    def __init__(self, input_size, output_embedding_size, embedding_layer_nonlinearity, dropout=0, use_separate_encoders=False, pre_concat=False, bypass_embed=False):
        super(BlockCountersEncoder, self).__init__()
        # Keep for reference
        self.input_size = input_size
        self.dropout = dropout
        self.use_separate_encoders = use_separate_encoders
        self.pre_concat = pre_concat
        self.bypass_embed = bypass_embed
        output_size = int(output_embedding_size/3)

        # Define layers
        if not self.bypass_embed:
            if not self.use_separate_encoders:
                self.embedding_layer = nn.Linear(input_size+3, output_size) if not self.pre_concat else nn.Linear(input_size*3, output_embedding_size)

                if embedding_layer_nonlinearity == 'relu':
                    self.embedding_layer_nonlinearity = nn.ReLU()
                elif embedding_layer_nonlinearity == 'tanh':
                    self.embedding_layer_nonlinearity = nn.Tanh()

                self.embed_dropout = nn.Dropout(p=dropout)

            else:
                self.placement_embed_layer = nn.Linear(input_size, output_size)
                self.next_placement_embed_layer = nn.Linear(input_size, output_size)
                self.next_removal_embed_layer = nn.Linear(input_size, output_size)

                if embedding_layer_nonlinearity == 'relu':
                    self.placement_embed_layer_nonlinearity = nn.ReLU()
                    self.next_placement_embed_layer_nonlinearity = nn.ReLU()
                    self.next_removal_embed_layer_nonlinearity = nn.ReLU()
                elif embedding_layer_nonlinearity == 'tanh':
                    self.placement_embed_layer_nonlinearity = nn.Tanh()
                    self.next_placement_embed_layer_nonlinearity = nn.Tanh()
                    self.next_removal_embed_layer_nonlinearity = nn.Tanh()

                self.embed_placement_dropout = nn.Dropout(p=dropout)
                self.embed_next_placement_dropout = nn.Dropout(p=dropout)
                self.embed_next_removal_dropout = nn.Dropout(p=dropout)

        self.input_encoding_size = output_embedding_size

        self.init_weights()

    def init_weights(self):
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def forward(self, encoder_inputs):
        all_placements_counter = to_var(encoder_inputs.block_counters['all_placements_counter'])
        all_next_placements_counter = to_var(encoder_inputs.block_counters['all_next_placements_counter'])
        all_next_removals_counter = to_var(encoder_inputs.block_counters['all_removals_counter'])

        if self.bypass_embed:
            final_encoding = torch.cat((all_placements_counter, all_next_placements_counter, all_next_removals_counter), dim=2)

        elif self.use_separate_encoders:
            placement_encoding = self.embed_placement_dropout(self.placement_embed_layer_nonlinearity(self.placement_embed_layer(all_placements_counter)))
            next_placement_encoding = self.embed_next_placement_dropout(self.next_placement_embed_layer_nonlinearity(self.next_placement_embed_layer(all_next_placements_counter)))
            next_removal_encoding = self.embed_next_removal_dropout(self.next_removal_embed_layer_nonlinearity(self.next_removal_embed_layer(all_next_removals_counter)))
            final_encoding = torch.cat((placement_encoding, next_placement_encoding, next_removal_encoding), dim=2)

        else:
            if self.pre_concat:
                final_encoding = torch.cat((all_placements_counter, all_next_placements_counter, all_next_removals_counter), dim=2)
                final_encoding = self.embed_dropout(self.embedding_layer_nonlinearity(self.embedding_layer(final_encoding)))
            else:
                placement_encoding = torch.cat((all_placements_counter, to_var(torch.Tensor([[[1,0,0]]]))), dim=2)
                next_placement_encoding = torch.cat((all_next_placements_counter, to_var(torch.Tensor([[[0,1,0]]]))), dim=2)
                next_removal_encoding = torch.cat((all_next_removals_counter, to_var(torch.Tensor([[[0,0,1]]]))), dim=2)

                placement_encoding = self.embed_dropout(self.embedding_layer_nonlinearity(self.embedding_layer(placement_encoding)))
                next_placement_encoding = self.embed_dropout(self.embedding_layer_nonlinearity(self.embedding_layer(next_placement_encoding)))
                next_removal_encoding = self.embed_dropout(self.embedding_layer_nonlinearity(self.embedding_layer(next_removal_encoding)))

                final_encoding = torch.cat((placement_encoding, next_placement_encoding, next_removal_encoding), dim=2)

        return EncoderContext(decoder_hidden=final_encoding, decoder_input_concat=final_encoding, decoder_hidden_concat=final_encoding, decoder_input_t0=final_encoding)

class BlockRegionCountersEncoder(nn.Module):
    """
        Regional block counters encoder
            - with an optional global block counters encoder as well (when used the network will use both encoders and combine them appropriately)
    """
    def __init__(self, input_size, output_embedding_size, embedding_layer_nonlinearity, dropout=0, use_separate_encoders=False, pre_concat=False, bypass_embed=False, use_global_counters=False, use_separate_global_embedding=False, global_counter_embedding_size=15, use_existing_blocks_counter=False):
        super(BlockRegionCountersEncoder, self).__init__()
        # Keep for reference
        self.dropout = dropout
        self.use_separate_encoders = use_separate_encoders
        self.pre_concat = pre_concat
        self.bypass_embed = bypass_embed
        self.use_global_counters = use_global_counters
        self.use_separate_global_embedding = use_separate_global_embedding
        self.global_counter_embedding_size = global_counter_embedding_size
        self.use_existing_blocks_counter = use_existing_blocks_counter

        input_size_for_global_counters = 0
        if self.use_global_counters and not self.use_separate_global_embedding:
            input_size_for_global_counters = 18

        self.input_size = input_size + input_size_for_global_counters

        # Define layers
        if not self.bypass_embed:
            if not self.use_separate_encoders:
                self.embedding_layer = nn.Linear(self.input_size, output_embedding_size)

                if embedding_layer_nonlinearity == 'relu':
                    self.embedding_layer_nonlinearity = nn.ReLU()
                elif embedding_layer_nonlinearity == 'tanh':
                    self.embedding_layer_nonlinearity = nn.Tanh()

                self.embed_dropout = nn.Dropout(p=dropout)

        if self.use_global_counters and self.use_separate_global_embedding:
            self.global_counters_encoder = BlockCountersEncoder(input_size=6, output_embedding_size=self.global_counter_embedding_size, embedding_layer_nonlinearity=embedding_layer_nonlinearity, dropout=self.dropout, pre_concat=True)

        self.input_encoding_size = output_embedding_size + (self.global_counter_embedding_size if self.use_global_counters and self.use_separate_global_embedding else 0)

        self.init_weights()

    def init_weights(self):
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def forward(self, encoder_inputs):
        block_counters_input = [to_var(encoder_inputs.last_action_bits)]
        region_block_counters = encoder_inputs.block_counters_spatial_tensors
        block_counters_input.extend([to_var(x) for x in list(sum(region_block_counters, ()))])
        final_encoding = torch.cat(block_counters_input, dim=2)

        if self.use_global_counters and not self.use_separate_global_embedding:
            all_placements_counter = to_var(encoder_inputs.block_counters['all_placements_counter'])
            all_next_placements_counter = to_var(encoder_inputs.block_counters['all_next_placements_counter'])
            all_next_removals_counter = to_var(encoder_inputs.block_counters['all_removals_counter'])
            global_counters_encoding = torch.cat((all_placements_counter, all_next_placements_counter, all_next_removals_counter), dim=2)
            final_encoding = torch.cat((global_counters_encoding, final_encoding), dim=2)

        final_encoding = self.embed_dropout(self.embedding_layer_nonlinearity(self.embedding_layer(final_encoding)))

        if self.use_global_counters and self.use_separate_global_embedding:
            global_counters_encoding = self.global_counters_encoder(encoder_inputs)
            final_encoding = torch.cat((global_counters_encoding.decoder_input_concat, final_encoding), dim=2)

        return EncoderContext(decoder_hidden=final_encoding, decoder_input_concat=final_encoding, decoder_hidden_concat=final_encoding, decoder_input_t0=final_encoding)
