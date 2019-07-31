import sys, torch, json, copy, pickle, re, os, numpy as np, pprint as pp, cProfile, pstats, io, traceback, itertools
if os.path.isdir('../../cwc-minecraft/build/install/Python_Examples/config_diff_tool'):
	# if running code from cwc-minecraft-models
	sys.path.append('../../cwc-minecraft/build/install/Python_Examples/config_diff_tool')
elif os.path.isdir('config_diff_tool'):
	# if running code from cwc-minecraft
	sys.path.append('config_diff_tool')
from diff import diff, get_diff, get_next_actions, build_region_specs, dict_to_tuple, is_feasible_next_placement
from diff_apps import get_type_distributions

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter

from utils import *
from vocab import Vocabulary
from dataset_filters import *
from plot_utils import plot_histogram

# MAIN CLASSES

class CwCDataset(Dataset):
	""" CwC Dataset compatible with torch.utils.data.DataLoader. """

	def __init__(
		self, model, split, lower=False, add_builder_utterances=False, compute_diff=True, compute_perspective=True,
		augment_dataset=False, augmentation_factor=0, exactly_k=False, strict=False,
		data_dir="../data/logs/", gold_configs_dir="../data/gold-configurations/", save_dest_dir="../data/saved_cwc_datasets", saved_dataset_dir="../data/saved_cwc_datasets/lower-no_diff/", vocab_dir="../vocabulary/",
		encoder_vocab=None, decoder_vocab=None, dump_dataset=False, load_dataset=False, transform=None, sample_filters = [],
		add_augmented_data=False, augmented_data_fraction=0.0, aug_data_dir="../data/augmented-no-spatial/logs/", aug_gold_configs_dir="../data/augmented-no-spatial/gold-configurations/"
	):
		"""
		Instantiates a dataset
			- If dump_dataset and load_dataset are both un-set, generates the dataset
			- If dump_dataset is set, also writes the generated dataset to file
			- If load_dataset is set, loads an existing dataset instead of generating (needed most often)

		By dataset, we mean self.samples and self.jsons -- the former being actual train/test examples, the latter being the json log files used to obtain these samples

		Args:
			model (string): model for which data loader is going to be used -- only used to selectively compute some stuff
			split (string): which of train/test/dev split to be used. If none, then reads and stores all data.
			lower: whether the data should be lowercased.
			add_builder_utterances: whether or not to obtain examples for builder utterances as well
			compute_diff: whether or not to compute the diff based representations
			compute_perspective: whether or not to compute the perspective coordinates based representations
			save_dest_dir: where to write the generated dataset
			saved_dataset_dir: where to load the saved dataset from
			encoder_vocab: encoder vocabulary wrapper.
			decoder_vocab: decoder vocabulary wrapper.
			dump_dataset: whether to generate dataset and write to file
			load_dataset: whether to load dataset from file
		"""

		self.model = model
		self.split = split
		self.lower = lower
		self.augmentation_factor = augmentation_factor
		self.add_builder_utterances = add_builder_utterances
		self.compute_diff = compute_diff
		self.compute_perspective = compute_perspective
		self.exactly_k = exactly_k
		self.strict = strict
		self.add_augmented_data = add_augmented_data
		self.augmented_data_fraction = augmented_data_fraction

		self.decoder_vocab = decoder_vocab
		self.encoder_vocab = encoder_vocab
		self.transform = transform

		self.num_prev_utterances = 1
		self.blocks_max_weight = 1
		self.use_builder_actions = False
		self.num_next_actions = 2
		self.include_empty_channel = False
		self.feasible_next_placements = False
		self.use_condensed_action_repr = False
		self.action_type_sensitive = False
		self.spatial_info_window_size = 1000
		self.counters_extra_feasibility_check = False
		self.use_existing_blocks_counter = False

		self.src_input_size_configs = x_range + y_range + z_range + len(type2id)
		self.src_input_size_next_actions = (x_range + y_range + z_range if not self.use_condensed_action_repr else 3) + len(type2id) + (len(action2id) if self.action_type_sensitive else 0)

		self.online_data = False # whether this if for architect demo or not aka online mode or not

		cwc_datasets_path = save_dest_dir

		lower_str = "lower" if self.lower else ""
		add_builder_utterances_str = "-add_builder_utterances" if self.add_builder_utterances else ""
		diff_str = '-no_diff' if not self.compute_diff else ""
		pers_str = '-no_perspective_coords' if not self.compute_perspective else ""
		aug_str = "-augmented" if self.add_augmented_data else ""

		# read augmented (synthetic) dataset
		if split == "train" and augment_dataset and augmentation_factor >= 1:
			# NOTE: ONLY FOR USE WITH LANGUAGE MODEL. DO NOT USE OTHERWISE!!

			split_str = "-" + self.split
			augmentation_factor_str = "-" + str(self.augmentation_factor)
			exactly_k_str = "-exactly_k" if self.exactly_k else ""
			strict_str = "-strict" if self.strict else ""
			augmented_data_filename = vocab_dir+"/synthetic_utterances" + split_str + "-" + lower_str + add_builder_utterances_str + augmentation_factor_str + exactly_k_str + strict_str + ".pkl"

			with open(augmented_data_filename, 'rb') as f:
				self.samples = pickle.load(f)

			print("Finished reading", augmented_data_filename)
			print("Loaded augmented dataset of size", len(self.samples))

		else:
			if load_dataset:
				dataset_dir = saved_dataset_dir

				print("Loading dataset ...\n")

				print("Loading self.samples ...")
				self.samples = load_pkl_data(dataset_dir + "/"+ self.split + "-samples.pkl")
				self.filter_augmented_samples()

				print("Loading self.jsons ...")
				self.jsons = load_pkl_data(dataset_dir + "/"+ self.split + "-jsons.pkl")

				print("Done! Loaded dataset of size", len(self.samples))

			else:
				self.jsons = list(
					map(
						remove_empty_states,
						map(
							reorder,
							get_logfiles_with_gold_config(data_dir, gold_configs_dir, split)
						)
					)
				) # TODO: Move the extra maps to a postprocesing step for the dataset?

				if self.add_augmented_data:
					print(timestamp(), "Adding augmented dataset...")

					def reformat_utterances(aug_observations_json):
						"""
							Joins tokens back with a space
						"""
						for world_state in aug_observations_json["WorldStates"]:
							world_state["ChatHistoryTokenized"] = list(map(
								lambda x: " ".join(x), world_state["ChatHistoryTokenized"]
							))
							world_state["ChatHistory"] = world_state.pop("ChatHistoryTokenized")

						return aug_observations_json

					self.jsons += list(
						map(
							remove_empty_states,
							map(
								reorder,
								map(
									reformat_utterances,
									get_logfiles_with_gold_config(aug_data_dir, aug_gold_configs_dir, split, from_aug_data=True)
								)
							)
						)
					)

				print(timestamp(), 'Started processing jsons to get samples...')
				self.samples = self.process_samples(lower, compute_diff=self.compute_diff, compute_perspective=self.compute_perspective)
				print(timestamp(), 'Done processing jsons to get samples.')

				print("Current dataset size", len(self.samples))
				print("Filtering...")
				for sample_filter in sample_filters:
					self.samples = list(filter(sample_filter, self.samples))

				print("Done! Loaded vanilla dataset of size", len(self.samples))

				if dump_dataset:
					sample_filters_names = list(map(lambda x: x.__name__, sample_filters))
					sample_filters_names = "-" + "-".join(sample_filters_names) if sample_filters_names else ""

					dataset_dir = lower_str + add_builder_utterances_str + diff_str + pers_str + aug_str + sample_filters_names
					dataset_dir = os.path.join(cwc_datasets_path, dataset_dir)

					if not os.path.exists(dataset_dir):
						os.makedirs(dataset_dir)

					print("Saving dataset ...\n")

					print("Saving self.jsons ...")
					save_pkl_data(dataset_dir + "/"+ self.split + "-jsons.pkl", self.jsons)
					save_pkl_data(dataset_dir + "/"+ self.split + "-jsons-2.pkl", self.jsons, protocol=2)

					print("Saving self.samples ...")
					save_pkl_data(dataset_dir + "/"+ self.split + "-samples.pkl", self.samples)

			self.augmentation_factor = 0

	def set_args(self, num_prev_utterances=1, blocks_max_weight=1, use_builder_actions=False, num_next_actions=2, include_empty_channel=False, use_condensed_action_repr=False, action_type_sensitive=False, feasible_next_placements=False, spatial_info_window_size=1000, counters_extra_feasibility_check=False, use_existing_blocks_counter=False):
		"""
			Selectively set some args in the object
		"""
		self.num_prev_utterances = num_prev_utterances
		self.blocks_max_weight = blocks_max_weight
		self.use_builder_actions = use_builder_actions
		self.num_next_actions = num_next_actions
		self.include_empty_channel = include_empty_channel
		self.feasible_next_placements = feasible_next_placements
		self.use_condensed_action_repr = use_condensed_action_repr
		self.action_type_sensitive = action_type_sensitive
		self.spatial_info_window_size = spatial_info_window_size
		self.counters_extra_feasibility_check = counters_extra_feasibility_check
		self.use_existing_blocks_counter = use_existing_blocks_counter
		self.src_input_size_next_actions = (x_range + y_range + z_range if not self.use_condensed_action_repr else 3) + len(type2id) + (len(action2id) if self.action_type_sensitive else 0)

	def get_sample(self, idx):
		""" Returns one data sample (utterance) in tokenized form. """
		return self.samples[idx]

	def filter_augmented_samples(self):
		samples = {'orig': [], 'aug': []}
		for sample in self.samples:
			samples['orig'].append(sample) if not sample.get('from_aug_data') else samples['aug'].append(sample)
		print('\nOriginal dataset contains', len(samples['orig']), 'original samples and', len(samples['aug']), 'augmented samples ('+str(len(samples['orig'])+len(samples['aug'])), 'total samples).')

		if self.augmented_data_fraction > 0 and len(samples['aug']) == 0:
			print('Error: you specified a fraction of augmented data, but the loaded dataset contains no augmented data.')
			sys.exit(0)

		if self.augmented_data_fraction == 0 and len(samples['aug']) == 0:
			return

		if self.augmented_data_fraction < 1.0:
			print('Filtering augmented samples with a fraction of', self.augmented_data_fraction, '...')
			chosen_aug_samples = np.random.choice(samples['aug'], int(self.augmented_data_fraction*len(samples['aug'])), replace=False)
			print('Randomly sampled', len(chosen_aug_samples), 'augmented samples from the full augmented set.')
			self.samples = samples['orig']
			self.samples.extend(chosen_aug_samples)

	def process_samples(self, lower, compute_diff=True, compute_perspective=True):
		""" Preprocesses the input JSONs and generates a list of data samples. """
		samples = []

		try:
			for j in range(len(self.jsons)):
				print("Processing json", j, "of", len(self.jsons))

				try:
					js = self.jsons[j]
					world_states = js["WorldStates"]
					final_observation = world_states[-1]
					gold_config = js["gold_config_structure"]

					last_world_state = None
					chat_history = []
					chat_with_actions_history = []

					gold_placements, gold_removals = get_gold_actions(world_states)

					for i in range(1, len(world_states)):
						observation = world_states[i]
						built_config = get_built_config(observation)
						builder_position = get_builder_position(observation)
						last_action = None
						gold_placement_list = gold_placements[i]
						gold_removal_list = gold_removals[i]

						for k, curr_world_state in enumerate(reversed(world_states[:i+1])):
							original_index = i-k

							# compare blocks with its prev world state
							curr_blocks = curr_world_state["BlocksInGrid"]
							prev_blocks = [] if original_index == 0 else world_states[original_index-1]["BlocksInGrid"]
							last_action = get_last_action(curr_blocks, prev_blocks)

							if last_action:
								break

						if not last_world_state:
							for i2 in range(len(observation["ChatHistory"])):
								chat_history.append(observation["ChatHistory"][i2].strip())

								for block in built_config:
									chat_with_actions_history.append({"idx": i, "action": "putdown", "type": block["type"], "built_config": built_config, "prev_config": None, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

								chat_with_actions_history.append({"idx": i, "action": "chat", "utterance": observation["ChatHistory"][i2].strip(), "built_config": built_config, "prev_config": None, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

						else:
							prev_config = get_built_config(last_world_state)
							config_diff = diff(gold_config=built_config, built_config=prev_config)
							delta = {"putdown": config_diff["gold_minus_built"], "pickup": config_diff["built_minus_gold"]}

							for action_type in delta:
								for block in delta[action_type]:
									chat_with_actions_history.append({"idx": i, "action": action_type, "type": block["type"], "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

							if len(observation["ChatHistory"]) > len(last_world_state["ChatHistory"]):
								for i3 in range(len(last_world_state["ChatHistory"]), len(observation["ChatHistory"])):
									chat_history.append(observation["ChatHistory"][i3].strip())
									chat_with_actions_history.append({"idx": i, "action": "chat", "utterance": observation["ChatHistory"][i3].strip(), "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

						last_world_state = observation

					# process dialogue line-by-line
					for i in range(len(chat_with_actions_history)):
						elem = chat_with_actions_history[i]

						if elem['action'] != 'chat':
							continue

						idx = elem['idx']
						line = elem['utterance']
						built_config = elem["built_config"]
						prev_config = elem["prev_config"]
						builder_position = elem["builder_position"]
						last_action = append_block_perspective_coords(builder_position, elem["last_action"])
						gold_placement_list = [append_block_perspective_coords(builder_position, block) for block in elem["gold_placement_list"]]
						gold_removal_list = [append_block_perspective_coords(builder_position, block) for block in elem["gold_removal_list"]]

						speaker = "Architect" if "Architect" in line.split()[0] else "Builder"
						if not self.add_builder_utterances and speaker == 'Builder':
							continue

						def valid_config(config):
							if not config:
								return True

							for block in config:
								x, y, z = block["x"]-x_min, block["y"]-y_min, block["z"]-z_min
								if x < 0 or x >= x_range or y < 0 or y >= y_range or z < 0 or z >= z_range:
									return False

							return True

						# temporary fix for troublesome configs
						if not valid_config(built_config) or not valid_config(prev_config):
							continue

						prefix = architect_prefix if speaker == "Architect" else builder_prefix
						next_utterance = line[len(prefix):]
						next_tokenized, _ = tokenize(next_utterance.lower() if lower else next_utterance)

						prev_utterances = []
						prev_utterances.append({'speaker': 'Builder', 'utterance': ['<dialogue>']})

						for k in range(i):
							prev_elem = chat_with_actions_history[k]

							if prev_elem['action'] != 'chat':
								prev_utterances.append({'speaker': 'Builder', 'utterance': ['<builder_'+prev_elem['action']+'_'+prev_elem['type']+'>']})

							else:
								prev_utterance = prev_elem['utterance']
								prev_speaker = "Architect" if "Architect" in prev_utterance.split()[0] else "Builder"
								prev_utterance = prev_utterance[len(architect_prefix):] if prev_speaker == 'Architect' else prev_utterance[len(builder_prefix):]
								prev_tokenized, _ = tokenize(prev_utterance.lower() if lower else prev_utterance)
								prev_utterances.append({'speaker': prev_speaker, 'utterance': prev_tokenized})

						# diff
						gold_v_built_diff, diffs_built_config_space, type_distributions_built_config_space, type_distributions_gold_config_space = None, None, None, None
						if compute_diff:
							gold_v_built_diff, perturbations_and_diffs = get_diff(gold_config=gold_config, built_config=built_config)

							# get type distributions
							diffs_built_config_space = list(map(lambda x: x.diff.diff_built_config_space, perturbations_and_diffs))
							type_distributions_built_config_space = reformat_type_distributions(
							  get_type_distributions(diffs_built_config_space=diffs_built_config_space, built_config=built_config)
							)

							# reverse diff
							_, perturbations_and_diffs_reverse = get_diff(gold_config=built_config, built_config=gold_config)

							# get type distributions
							diffs_gold_config_space = list(map(lambda x: x.diff.diff_built_config_space, perturbations_and_diffs_reverse))
							type_distributions_gold_config_space = reformat_type_distributions(
							  get_type_distributions(diffs_built_config_space=diffs_gold_config_space, built_config=gold_config)
							)

						perspective_coordinates = None if not compute_perspective else get_perspective_coord_repr(builder_position)

						samples.append(
							{
								'next_speaker': speaker, # architect or builder
								'next_utterance': next_tokenized, # utterance to be predicted
								'prev_utterances': prev_utterances, # previous utterances
								'gold_config': gold_config,
								'built_config': built_config,
								'diff': gold_v_built_diff, # diff based on one single optimal alignment
								'last_action': last_action, # last block placement action
								'gold_placement_list': gold_placement_list,
								'gold_removal_list': gold_removal_list,
								'builder_position': builder_position,
								'perspective_coordinates': perspective_coordinates,
								'type_distributions_built_config_space': type_distributions_built_config_space,
								'type_distributions_gold_config_space': type_distributions_gold_config_space,
								'from_aug_data': js['from_aug_data'],
								'diffs_built_config_space': diffs_built_config_space, # all diffs based on all optimal alignments -- in the built config space
								'json_id': j, # ID of the json this sample was obtained from
								'sample_id': idx, # ID assigned to this sample
								'next_utterance_raw': next_utterance # raw format of the utterance to be predicted -- for downstream purposes
							} # NOTE: data format of a sample
						)

				except Exception:
					print('Something went wrong processing this json, skipping...')
					traceback.print_exc()
					sys.exit(0)

		except KeyboardInterrupt:
			print('Exiting from processing json early... Not all samples have been added.')

		return samples

	def __len__(self):
		""" Returns length of dataset. """
		return len(self.samples)

	def __getitem__(self, idx):
		""" Computes the tensor representations of a sample """
		sample = self.samples[idx]

		# Convert utterance (string) to word IDs.
		next_tokens = sample["next_utterance"]

		next_utterance_input = []
		next_utterance_input.append(self.decoder_vocab('<architect>')) # NOTE: no need to change for LM using builder utterances too
		next_utterance_input.extend([self.decoder_vocab(token) for token in next_tokens])

		next_utterance_output = []
		next_utterance_output.extend([self.decoder_vocab(token) for token in next_tokens])
		next_utterance_output.append(self.decoder_vocab('</architect>')) # NOTE: no need to change for LM using builder utterances too

		i = 0
		utterances_idx = len(sample["prev_utterances"])-1
		utterances_to_add = []
		prev_utterances = []

		while i < self.num_prev_utterances:
			if utterances_idx < 0:
				break

			prev = sample["prev_utterances"][utterances_idx]
			speaker = prev["speaker"]
			utterance = prev["utterance"]

			if "<builder_" in utterance[0]:
				if self.use_builder_actions:
					utterances_to_add.insert(0, prev)
				i -= 1

			elif "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
				i -= 1

			else:
				utterances_to_add.insert(0, prev)

			utterances_idx -= 1
			i += 1

		if self.online_data:
			# use only one previous utterance for architect demo
			utterances_to_add = [utterances_to_add[-1]]

		for prev in utterances_to_add:
			speaker = prev["speaker"]
			utterance = prev["utterance"]

			if "<dialogue>" in utterance[0]:
				prev_utterances.append(self.encoder_vocab('<dialogue>'))

			elif "<builder_" in utterance[0]:
				if self.use_builder_actions:
					prev_utterances.append(self.encoder_vocab(utterance[0]))
				i -= 1

			else:
				start_token = self.encoder_vocab('<architect>') if 'Architect' in speaker else self.encoder_vocab('<builder>')
				end_token = self.encoder_vocab('</architect>') if 'Architect' in speaker else self.encoder_vocab('</builder>')
				prev_utterances.append(start_token)
				prev_utterances.extend(self.encoder_vocab(token) for token in utterance)
				prev_utterances.append(end_token)

		# temporary fix: floats in configs
		for config_type in ['built_config', 'gold_config']:
			config = sample[config_type]
			for block in config:
				for key in ['x', 'y', 'z']:
					block[key] = int(block[key])

		# built config
		built_config = sample["built_config"]
		built_config_repr = get_one_hot_repr(built_config) if self.model == 'seq2seq_world_state' else None
		built_config_3d_repr = get_3d_repr(built_config, max_weight=self.blocks_max_weight, include_empty_channel=self.include_empty_channel) if self.model == 'cnn_3d' else None

		# gold config
		gold_config = sample["gold_config"] # NOTE: already sorted by layer
		gold_config_repr = get_one_hot_repr(gold_config) if self.model == 'seq2seq_world_state' else None
		gold_config_3d_repr = get_3d_repr(gold_config, include_empty_channel=self.include_empty_channel) if self.model == 'cnn_3d' else None

		perspective_coord_repr = None
		if isinstance(sample["perspective_coordinates"], np.ndarray):
			perspective_coord_repr = torch.from_numpy(sample["perspective_coordinates"]).type(torch.FloatTensor)

		type_distributions_built_config_space = sample['type_distributions_built_config_space']
		type_distributions_gold_config_space = sample['type_distributions_gold_config_space']

		built_config_type_dist, gold_config_type_dist = None, None

		if isinstance(type_distributions_built_config_space, np.ndarray):
			if not self.include_empty_channel:
				type_distributions_built_config_space = type_distributions_built_config_space[:-1][:][:][:]
				type_distributions_gold_config_space = type_distributions_gold_config_space[:-1][:][:][:]

			built_config_type_dist = torch.from_numpy(type_distributions_built_config_space).type(torch.FloatTensor)
			gold_config_type_dist = torch.from_numpy(type_distributions_gold_config_space).type(torch.FloatTensor)

		# diff
		diff = sample["diff"]

		# last action
		last_action = sample["last_action"]

		next_actions_gold = {"gold_minus_built": sample["gold_placement_list"][:int(self.num_next_actions/2)], "built_minus_gold": sample["gold_removal_list"][:int(self.num_next_actions/2)]}
		next_actions_gold_repr = get_next_actions_repr(next_actions_gold, last_action, action_type_sensitive=self.action_type_sensitive, use_condensed_action_repr=self.use_condensed_action_repr)

		# next actions
		next_actions, next_actions_repr = None, None
		builder_position = sample['builder_position']
		if diff and self.model == 'utterances_and_next_actions':
			next_actions = get_next_actions(all_next_actions=diff, num_next_actions_needed=self.num_next_actions, last_action=last_action, built_config=built_config, feasible_next_placements=self.feasible_next_placements)
			next_actions['gold_minus_built'] = [append_block_perspective_coords(builder_position, block) for block in next_actions['gold_minus_built']]
			next_actions['built_minus_gold'] = [append_block_perspective_coords(builder_position, block) for block in next_actions['built_minus_gold']]
			next_actions_repr = get_next_actions_repr(next_actions, last_action, action_type_sensitive=self.action_type_sensitive, use_condensed_action_repr=self.use_condensed_action_repr)

		# block global counters
		diffs_built_config_space = sample["diffs_built_config_space"]
		block_counters = get_block_counters(diffs_built_config_space, built_config=built_config, built_config_in_region=built_config, extra_check=self.counters_extra_feasibility_check) if diff else None

		# block region counters
		block_counters_spatial_info = get_block_counters_spatial_info(diffs_built_config_space, built_config, last_action, builder_position, window_size=self.spatial_info_window_size, extra_check=self.counters_extra_feasibility_check) if diff else None
		block_counters_spatial_tensors = [] # FIXME: UPDATE WHEN MORE REGIONS ARE CONSIDERED
		if block_counters_spatial_info:
			if self.use_existing_blocks_counter:
				block_counters_spatial_tensors = [(torch.Tensor(x.block_counters.all_placements_counter), torch.Tensor(x.block_counters.all_next_placements_counter), torch.Tensor(x.block_counters.all_removals_counter), torch.Tensor(x.block_counters.all_existing_blocks_counter)) for x in block_counters_spatial_info]
			else:
				block_counters_spatial_tensors = [(torch.Tensor(x.block_counters.all_placements_counter), torch.Tensor(x.block_counters.all_next_placements_counter), torch.Tensor(x.block_counters.all_removals_counter)) for x in block_counters_spatial_info]

		last_action_bit = [[1]] if not last_action else [[0]]

		# pp.pprint(list(map(lambda x: x.__dict__, block_counters_spatial_info)))
		# print(block_counters_spatial_tensors)
		# print("\n\n\n\n")

		# print("get_item")
		# pp.PrettyPrinter(indent=4).pprint(prev_utterances)
		# print(sorted(type2id.keys()))
		# print(block_counters.all_placements_counter[0])
		# print(block_counters.all_removals_counter[0])
		from operator import add
		all_actions = list( map(add, block_counters.all_placements_counter[0], block_counters.all_removals_counter[0]) )
		colors_to_all_actions = dict(zip(sorted(type2id.keys()), all_actions))

		return (
			torch.Tensor(prev_utterances),
			torch.tensor(built_config_repr) if built_config_repr else None,
			torch.tensor(gold_config_repr) if gold_config_repr else None,
			torch.Tensor(next_utterance_input), # utterance to be predicted -- formatted for decoder inputs
			torch.Tensor(next_utterance_output), # utterance to be predicted -- formatted for decoder outputs
			torch.Tensor(next_actions_repr["next_placements_repr"]) if next_actions_repr else None,
			torch.Tensor(next_actions_repr["next_removals_repr"]) if next_actions_repr else None,
			torch.Tensor(next_actions_gold_repr["next_placements_repr"]),
			torch.Tensor(next_actions_gold_repr["next_removals_repr"]),
			torch.Tensor(block_counters.all_placements_counter) if block_counters else None, # global block counters
			torch.Tensor(block_counters.all_next_placements_counter) if block_counters else None, # global block counters
			torch.Tensor(block_counters.all_removals_counter) if block_counters else None, # global block counters
			block_counters_spatial_tensors, # regional block counters
			torch.Tensor(last_action_bit), # encoding of last action
			RawInputs(next_actions, next_actions_gold, json_id=sample.get('json_id'), sample_id=sample.get('sample_id'), next_utterance_raw=sample.get('next_utterance_raw'), built_config_ss=built_config, gold_config_ss=gold_config, colors_to_all_actions=colors_to_all_actions), # raw inputs for downstream purposes
			built_config_3d_repr,
			gold_config_3d_repr,
			perspective_coord_repr,
			built_config_type_dist,
			gold_config_type_dist
		) # NOTE: data format of an item

	def collate_fn(self, data):
		"""Creates a mini-batch of items (batch size = 1 for now)

		Returns:
			A tuple of the following:
				- inputs to the encoder
				- ground truth inputs to the decoder RNN
				- ground truth outputs for the decoder RNN
				- some inputs in raw format for downstream use cases
		"""
		def merge_text(sequences):
			lengths = [len(seq) for seq in sequences]
			padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
			for i, seq in enumerate(sequences):
				end = lengths[i]
				padded_seqs[i, :end] = seq[:end]
			return padded_seqs, lengths

		def merge_configs(sequences):
			if not isinstance(sequences[0], torch.Tensor) and not sequences[0]:
				return None, None

			lengths = [len(seq) for seq in sequences]
			padded_seqs = torch.zeros(len(sequences), max(lengths), sequences[0].size()[1])
			for i, seq in enumerate(sequences):
				end = lengths[i]
				padded_seqs[i, :end, :] = seq
			return padded_seqs, lengths

		def stack_reprs(reprs):
			if not isinstance(reprs[0], torch.Tensor) and not reprs[0]:
				return None
			return torch.stack(reprs, 0)

		# Sort a data list by utterance length in descending order.
		prev_utterances, built_configs, gold_configs, target_inputs, target_outputs, next_placements, next_removals, gold_placements, gold_removals, all_placements_counter, all_next_placements_counter, all_removals_counter, block_counters_spatial_tensors, last_action_bits, raw_inputs, built_configs_3d, gold_configs_3d, perspective_coord_reprs, built_config_type_dists, gold_config_type_dists = zip(*data)

		prev_utterances, prev_utterances_lengths = merge_text(prev_utterances)
		built_configs, built_config_lengths = merge_configs(built_configs)
		gold_configs, gold_config_lengths = merge_configs(gold_configs)

		next_placements, next_placements_lengths = merge_configs(next_placements)
		next_removals, next_removals_lengths = merge_configs(next_removals)
		gold_placements, gold_placements_lengths = merge_configs(gold_placements)
		gold_removals, gold_removals_lengths = merge_configs(gold_removals)

		all_placements_counter = stack_reprs(all_placements_counter)
		all_next_placements_counter = stack_reprs(all_next_placements_counter)
		all_removals_counter = stack_reprs(all_removals_counter)

		next_actions = {
			"next_placements": next_placements,
			"next_removals": next_removals
		}
		next_actions_lengths = {
			"next_placements_lengths": next_placements_lengths,
			"next_removals_lengths": next_removals_lengths
		}

		gold_actions = {
			"gold_placements": gold_placements,
			"gold_removals": gold_removals
		}
		gold_actions_lengths = {
			"gold_placements_lengths": gold_placements_lengths,
			"gold_removals_lengths": gold_removals_lengths
		}

		block_counters = {
			"all_placements_counter": all_placements_counter,
			"all_next_placements_counter": all_next_placements_counter,
			"all_removals_counter": all_removals_counter
		}

		block_counters_spatial_tensors = block_counters_spatial_tensors[0]
		if self.use_existing_blocks_counter:
			block_counters_spatial_tensors = [(w.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)) for (w,x,y,z) in block_counters_spatial_tensors]
		else:
			block_counters_spatial_tensors = [(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)) for (x,y,z) in block_counters_spatial_tensors]

		last_action_bits = stack_reprs(last_action_bits)
		built_configs_3d = stack_reprs(built_configs_3d)
		gold_configs_3d = stack_reprs(gold_configs_3d)
		perspective_coord_reprs = stack_reprs(perspective_coord_reprs)
		built_config_type_dists = stack_reprs(built_config_type_dists)
		gold_config_type_dists = stack_reprs(gold_config_type_dists)

		target_inputs, target_lengths = merge_text(target_inputs)
		target_outputs, target_lengths = merge_text(target_outputs)

		raw_inputs = raw_inputs[0]

		return (
			EncoderInputs(prev_utterances, prev_utterances_lengths, built_configs, built_config_lengths, gold_configs, gold_config_lengths, next_actions, next_actions_lengths, gold_actions, gold_actions_lengths, block_counters, block_counters_spatial_tensors, last_action_bits, built_configs_3d, gold_configs_3d, perspective_coord_reprs, built_config_type_dists, gold_config_type_dists),
			DecoderInputs(target_inputs, target_lengths),
			DecoderOutputs(target_outputs, target_lengths),
			raw_inputs
		)

	def get_data_loader(self, batch_size=1, shuffle=True, num_workers=1):
		# Data loader for CwC Dataset.
		# This will return (targets, lengths) for every iteration.
		# targets: torch tensor of shape (batch_size, padded_length).
		# lengths: list of valid lengths for each padded utterance, sorted in descending order. Length is (batch_size).
		return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

	# FIXME: Move this to a dedicated data augmentation part
	def augment_dataset(self, k, vocab_dir="../vocabulary/"):
		# data structures to hold info on the data augmentation
		self.word_counts_augmentation = defaultdict(int)
		self.tokenized_data_augmentation = []

		# generate synthetic utterances -- update above data structures
		for original_sample in self.samples:
			tokenized_utterance = original_sample["next_utterance"]
			# get new examples
			new_examples = self.get_new_examples(tokenized_utterance, k)
			# add to dataset
			self.tokenized_data_augmentation += new_examples
			for example in new_examples:
				for word in example:
					self.word_counts_augmentation[word] += 1

		# augment original dataset with the synthetic utterances
		for tokenized_synthetic_utterance in self.tokenized_data_augmentation:
			synthetic_sample = {
				'next_speaker': self.samples[0]['next_speaker'], # dummy value
				'next_utterance': tokenized_synthetic_utterance,
				'prev_utterances': self.samples[0]['prev_utterances'], # dummy value
				'gold_config': self.samples[0]['gold_config'], # dummy value
				'built_config': self.samples[0]['built_config'] # dummy value
			}
			self.samples.append(synthetic_sample)

		# write synthetic utterances to file
		print("Writing synthetic utterances to file...")
		split = "-" + self.split
		lower = "-lower" if self.lower else ""
		add_builder_utterances = "-add_builder_utterances" if self.add_builder_utterances else ""
		augmentation_factor = "-" + str(self.augmentation_factor)
		with open(vocab_dir+"/synthetic_utterances" + split + lower + add_builder_utterances + augmentation_factor + ".txt", 'w') as f:
			for tokenized_utterance in self.tokenized_data_augmentation:
				to_write = pprint.pformat(tokenized_utterance) + "\n\n"
				f.write(to_write)

		print("Done writing!")

	# FIXME: Move this to a dedicated data augmentation part
	def get_new_examples(self, tokenized_utterance, k):
		def f(token):
			# map token to list of all possible substitutions
			token_substitutions = [token]
			if token in self.substitutions:
				token_substitutions += self.substitutions[token]
			return token_substitutions

		# map each token to a list of it's substitutions including itself
		substitutions_list = list(map(f, tokenized_utterance))

		# generate all possible combinations -- cartesian product of a 2d list
		samples_list = list(map(lambda x: np.random.choice(x, size=k, replace=True).tolist(), substitutions_list))
		new_examples = list(map(list, [*zip(*samples_list)]))

		# filter out duplicate examples
		new_examples = list(filter(lambda x: x != tokenized_utterance, new_examples)) # filter out original utterance
		new_examples = [list(x) for x in set(tuple(x) for x in new_examples)] # select only unique new synthetic utterances

		return new_examples

	def printCoords(self):
		print(self.dpxs_placement)

class EncoderInputs:
	def __init__(self, prev_utterances, prev_utterances_lengths, built_configs, built_config_lengths, gold_configs, gold_config_lengths, next_actions, next_actions_lengths, gold_actions, gold_actions_lengths, block_counters, block_counters_spatial_tensors, last_action_bits, built_configs_3d, gold_configs_3d, perspective_coord_reprs, built_config_type_dists, gold_config_type_dists):
		self.prev_utterances = prev_utterances # previous utterances
		self.prev_utterances_lengths = prev_utterances_lengths
		self.built_configs = built_configs # built config
		self.built_config_lengths = built_config_lengths
		self.gold_configs = gold_configs # gold config
		self.gold_config_lengths = gold_config_lengths
		self.next_actions = next_actions
		self.next_actions_lengths = next_actions_lengths
		self.gold_actions = gold_actions
		self.gold_actions_lengths = gold_actions_lengths
		self.block_counters = block_counters # global block counters
		self.block_counters_spatial_tensors = block_counters_spatial_tensors # regional block counters
		self.last_action_bits = last_action_bits # last action encoding
		self.built_configs_3d = built_configs_3d
		self.gold_configs_3d = gold_configs_3d
		self.perspective_coord_reprs = perspective_coord_reprs
		self.built_config_type_dists = built_config_type_dists
		self.gold_config_type_dists = gold_config_type_dists

class DecoderInputs:
	def __init__(self, target_inputs, target_lengths, target_inputs_neg=None, target_lengths_neg=None):
		self.target_inputs = target_inputs # ground truth inputs for decoder RNN
		self.target_lengths = target_lengths
		self.target_inputs_neg = target_inputs_neg
		self.target_lengths_neg = target_lengths_neg

class DecoderOutputs:
	def __init__(self, target_outputs, target_lengths, target_outputs_neg=None, target_lengths_neg=None):
		self.target_outputs = target_outputs # ground truth outputs for decoder RNN
		self.target_lengths = target_lengths
		self.target_outputs_neg = target_outputs_neg
		self.target_lengths_neg = target_lengths_neg

class RawInputs:
	"""
		Raw representations of various inputs for downstream use cases
	"""

	def __init__(self, next_actions_raw, gold_next_actions_raw, json_id=None, sample_id=None, next_utterance_raw=None, built_config_ss=None, gold_config_ss=None, colors_to_all_actions=None):
		self.next_actions_raw = next_actions_raw
		self.gold_next_actions_raw = gold_next_actions_raw
		self.json_id = json_id # json id of the game log from where this train/test example was obtained
		self.sample_id = sample_id # sample id of the train/test example
		self.next_utterance_raw = next_utterance_raw # next utterance to be predicted
		self.built_config_ss = built_config_ss
		self.gold_config_ss = gold_config_ss
		self.colors_to_all_actions = colors_to_all_actions

# UTILS

class Region:
	"""
		Stores a specfic region in 3d space
	"""
	def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, block_counters=None, region_id=None):
		"""
			- Bounds of the region
			- Block counters for the region
			- A unique ID based on whether the region is left, right, etc. of the last action
		"""
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max
		self.z_min = z_min
		self.z_max = z_max

		assert self.x_min <= self.x_max and self.y_min <= self.y_max and self.z_min <= self.z_max, "Invalid x/y/z bounds for Region object."

		self.block_counters = block_counters
		self.region_id = region_id

	def set_block_counters(self, diffs_built_config_space, built_config, extra_check):
		"""
			Compute and set block counters for region
		"""

		# filter actions in diffs to only include actions within the region
		region_diffs = list(map(lambda x: self.get_region_diff(x), diffs_built_config_space))

		# filter blocks in built config to only include blocks within the region
		built_config_in_region = list(filter(lambda x: self.is_in_region(x), built_config))

		# get block counters
		self.block_counters = get_block_counters(region_diffs, built_config=built_config, built_config_in_region=built_config_in_region, extra_check=extra_check)

	def get_region_diff(self, diff):
		"""
			Reduce a diff to being specific to the region
		"""

		all_placements = diff["gold_minus_built"]
		all_removals = diff["built_minus_gold"]

		placements_in_region = list(filter(lambda x: self.is_in_region(x), all_placements))
		removals_in_region = list(filter(lambda x: self.is_in_region(x), all_removals))

		region_diff = {
			"gold_minus_built": placements_in_region,
			"built_minus_gold": removals_in_region
		}

		return region_diff

	def is_in_region(self, action):
		"""
			Check if an action is within the region or not
		"""

		return action["x"] in range(self.x_min, self.x_max + 1) and \
		action["y"] in range(self.y_min, self.y_max + 1) and \
		action["z"] in range(self.z_min, self.z_max + 1)

	def get_region_id(self, builder_position, last_action):
		"""
			Compute the unique ID for the region based on whether it is left, right, etc. of the last action
		"""
		# discretize builder's yaw into the 4 canonical directions
		builder_yaw = builder_position["yaw"]
		builder_yaw_discrete = discretize_yaw(builder_yaw)

		# given a canonical yaw direction and delta vectors wrt that direction, infer which cell is left, which is right and so on
		diff_vector = {
			"x": (self.x_max + self.x_min) / 2 - last_action["x"], # diff using mean of region
			"y": (self.y_max + self.y_min) / 2 - last_action["y"],
			"z": (self.z_max + self.z_min) / 2 - last_action["z"]
		}

		# infer what is left, right, etc. based on canonical yaw direction
		if builder_yaw_discrete == 0:
			diff_vector_to_direction = {
				"+x": "left",
				"-x": "right",
				"+z": "front",
				"-z": "back"
			}
		elif builder_yaw_discrete == 90:
			diff_vector_to_direction = {
				"+x": "back",
				"-x": "front",
				"+z": "left",
				"-z": "right"
			}
		elif builder_yaw_discrete == 180:
			diff_vector_to_direction = {
				"+x": "right",
				"-x": "left",
				"+z": "back",
				"-z": "front"
			}
		elif builder_yaw_discrete == -90:
			diff_vector_to_direction = {
				"+x": "front",
				"-x": "back",
				"+z": "right",
				"-z": "left"
			}

		diff_vector_to_direction["+y"] = "top"
		diff_vector_to_direction["-y"] = "down"

		# convert diff vector to one left, right, etc. and then convert to a unique id

		# when last action cell itself
		if diff_vector["x"] == 0 and diff_vector["y"] == 0 and diff_vector["z"] == 0:
			self.region_id = direction_to_id["null"]

		# when adjacent cells or rows/columns
		if diff_vector["x"] != 0 and diff_vector["y"] == 0 and diff_vector["z"] == 0:
			if diff_vector["x"] > 0:
				if diff_vector["x"] == 1:
					self.region_id = direction_to_id[diff_vector_to_direction["+x"]]
				else:
					self.region_id = direction_to_id[diff_vector_to_direction["+x"] + "_row"]
			else:
				if diff_vector["x"] == -1:
					self.region_id = direction_to_id[diff_vector_to_direction["-x"]]
				else:
					self.region_id = direction_to_id[diff_vector_to_direction["-x"] + "_row"]
		elif diff_vector["x"] == 0 and diff_vector["y"] != 0 and diff_vector["z"] == 0:
			if diff_vector["y"] > 0:
				if diff_vector["y"] == 1:
					self.region_id = direction_to_id["top"]
				else:
					self.region_id = direction_to_id["top_column"]
			else:
				if diff_vector["y"] == -1:
					self.region_id = direction_to_id["down"]
				else:
					self.region_id = direction_to_id["down_column"]
		elif diff_vector["x"] == 0 and diff_vector["y"] == 0 and diff_vector["z"] != 0:
			if diff_vector["z"] > 0:
				if diff_vector["z"] == 1:
					self.region_id = direction_to_id[diff_vector_to_direction["+z"]]
				else:
					self.region_id = direction_to_id[diff_vector_to_direction["+z"] + "_row"]
			else:
				if diff_vector["z"] == -1:
					self.region_id = direction_to_id[diff_vector_to_direction["-z"]]
				else:
					self.region_id = direction_to_id[diff_vector_to_direction["-z"] + "_row"]

		# when adjacent quadrants
		if diff_vector["x"] != 0 and diff_vector["y"] != 0 and diff_vector["z"] == 0:
			signed_x = "+x" if diff_vector["x"] > 0 else "-x"
			signed_y = "+y" if diff_vector["y"] > 0 else "-y"
			self.region_id = direction_to_id[
				stringify((diff_vector_to_direction[signed_x], diff_vector_to_direction[signed_y]))
			]
		elif diff_vector["x"] == 0 and diff_vector["y"] != 0 and diff_vector["z"] != 0:
			signed_y = "+y" if diff_vector["y"] > 0 else "-y"
			signed_z = "+z" if diff_vector["z"] > 0 else "-z"
			self.region_id = direction_to_id[
				stringify((diff_vector_to_direction[signed_y], diff_vector_to_direction[signed_z]))
			]
		elif diff_vector["x"] != 0 and diff_vector["y"] == 0 and diff_vector["z"] != 0:
			signed_z = "+z" if diff_vector["z"] > 0 else "-z"
			signed_x = "+x" if diff_vector["x"] > 0 else "-x"
			self.region_id = direction_to_id[
				stringify((diff_vector_to_direction[signed_z], diff_vector_to_direction[signed_x]))
			]

		# when adjacent octants
		if diff_vector["x"] != 0 and diff_vector["y"] != 0 and diff_vector["z"] != 0:
			signed_x = "+x" if diff_vector["x"] > 0 else "-x"
			signed_y = "+y" if diff_vector["y"] > 0 else "-y"
			signed_z = "+z" if diff_vector["z"] > 0 else "-z"
			self.region_id = direction_to_id[
				stringify((diff_vector_to_direction[signed_x], diff_vector_to_direction[signed_y], diff_vector_to_direction[signed_z]))
			]

		return self.region_id

# obtain mapping of relative directions to ID

# for last action cell itself
null = "null"

# for adjacent cells
lr = ["left", "right"]
td = ["top", "down"]
fb = ["front", "back"]

# for adjacent rows/columns
lr_rows = list(map(lambda x: x + "_row", lr))
td_columns = list(map(lambda x: x + "_column", td))
fb_rows = list(map(lambda x: x + "_row", fb))

# for adjacent quadrants
def stringify(directions):
	"""
		Converts a bunch of directions into a unique identifier string -- irrespective of how directions are ordered in the iterable
		NOTE: DO NOT CHANGE THIS LOGIC WITHOUT THOUGHT
	"""
	return "_".join(sorted(list(directions)))

lr_td = list(map(stringify, list(itertools.product(lr, td))))
td_fb = list(map(stringify, list(itertools.product(td, fb))))
fb_lr = list(map(stringify, list(itertools.product(fb, lr))))

# for adjacent octants
lr_td_fb = list(map(stringify, list(itertools.product(lr, td, fb))))

# unify all and get a map
all_directions = [null] + lr + td + fb + lr_rows + td_columns + fb_rows + lr_td + td_fb + fb_lr + lr_td_fb # NOTE: DO NOT CHANGE THIS ORDERING WITHOUT THOUGHT!
direction_to_id = {k: v for v, k in enumerate(all_directions)}

def discretize_yaw(yaw):
	"""
		Discretize a yaw angle into the 4 canonical yaw angles/directions
	"""
	# normalize to [0, 360]
	if yaw < 0:
		yaw_normalized = 360 + yaw
	else:
		yaw_normalized = yaw

	# discretize
	if (yaw_normalized >= 270 + 45 and yaw_normalized <= 360) or (yaw_normalized >= 0 and yaw_normalized < 0 + 45):
		return 0
	elif yaw_normalized >= 0 + 45 and yaw_normalized < 90 + 45:
		return 90
	elif yaw_normalized >= 90 + 45 and yaw_normalized < 180 + 45:
		return 180
	else:
		return -90

def get_block_counters_spatial_info(diffs_built_config_space, built_config, last_action, builder_position, window_size, extra_check):
	"""
		Obtain block counters based spatial info
	"""
	# degenerate case
	if not last_action:
		last_action = {
			"x": 0,
			"y": 1,
			"z": 0
		}

	# obtain regions adjacent to last action
	adjacent_regions = get_adjacent_regions(last_action, window_size)

	# get counters for each region
	list(
		map(
			lambda x: x.set_block_counters(diffs_built_config_space, built_config, extra_check), # mutating
			adjacent_regions
		)
	)

	# obtain canonical ordering of regions -- based on directions
	adjacent_regions = sorted(adjacent_regions, key = lambda x: x.get_region_id(builder_position, last_action)) # mutating

	return adjacent_regions

def get_adjacent_regions(action, window_size):
	"""
		Returns a list of the 6 adjacent cells + 6 adjacent rows/columns
	"""
	assert window_size >= 1, "Spatial info window size < 1 is not supported."

	action_cell = Region(x_min = action["x"], x_max = action["x"], y_min = action["y"], y_max = action["y"], z_min = action["z"], z_max = action["z"])

	if window_size >= 1:
		cells = [
			Region(x_min = action["x"] + 1, x_max = action["x"] + 1, y_min = action["y"], y_max = action["y"], z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"] - 1, x_max = action["x"] - 1, y_min = action["y"], y_max = action["y"], z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"] + 1, y_max = action["y"] + 1, z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"] - 1, y_max = action["y"] - 1, z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"], y_max = action["y"], z_min = action["z"] + 1, z_max = action["z"] + 1),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"], y_max = action["y"], z_min = action["z"] - 1, z_max = action["z"] - 1)
		]

		quadrants = []

		for sign_x in [1, -1]:
			for sign_y in [1, -1]:
				quadrants.append(
					Region(
						x_min = action["x"] + 1 if sign_x == 1 else action["x"] - window_size,
						x_max = action["x"] + window_size if sign_x == 1 else action["x"] - 1,
						y_min = action["y"] + 1 if sign_y == 1 else action["y"] - window_size,
						y_max = action["y"] + window_size if sign_y == 1 else action["y"] - 1,
						z_min = action["z"],
						z_max = action["z"]
					)
				)

		for sign_y in [1, -1]:
			for sign_z in [1, -1]:
				quadrants.append(
					Region(
						x_min = action["x"],
						x_max = action["x"],
						y_min = action["y"] + 1 if sign_y == 1 else action["y"] - window_size,
						y_max = action["y"] + window_size if sign_y == 1 else action["y"] - 1,
						z_min = action["z"] + 1 if sign_z == 1 else action["z"] - window_size,
						z_max = action["z"] + window_size if sign_z == 1 else action["z"] - 1
					)
				)

		for sign_z in [1, -1]:
			for sign_x in [1, -1]:
				quadrants.append(
					Region(
						x_min = action["x"] + 1 if sign_x == 1 else action["x"] - window_size,
						x_max = action["x"] + window_size if sign_x == 1 else action["x"] - 1,
						y_min = action["y"],
						y_max = action["y"],
						z_min = action["z"] + 1 if sign_z == 1 else action["z"] - window_size,
						z_max = action["z"] + window_size if sign_z == 1 else action["z"] - 1
					)
				)

		octants = []

		for sign_x in [1, -1]:
			for sign_y in [1, -1]:
				for sign_z in [1, -1]:
					octants.append(
						Region(
							x_min = action["x"] + 1 if sign_x == 1 else action["x"] - window_size,
							x_max = action["x"] + window_size if sign_x == 1 else action["x"] - 1,
							y_min = action["y"] + 1 if sign_y == 1 else action["y"] - window_size,
							y_max = action["y"] + window_size if sign_y == 1 else action["y"] - 1,
							z_min = action["z"] + 1 if sign_z == 1 else action["z"] - window_size,
							z_max = action["z"] + window_size if sign_z == 1 else action["z"] - 1
						)
					)

	else:
		cells = []
		quadrants = []
		octants = []

	if window_size >= 2:
		rows_and_columns = [
			Region(x_min = action["x"] + 2, x_max = action["x"] + window_size, y_min = action["y"], y_max = action["y"], z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"] - window_size, x_max = action["x"] - 2, y_min = action["y"], y_max = action["y"], z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"] + 2, y_max = action["y"] + window_size, z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"] - window_size, y_max = action["y"] - 2, z_min = action["z"], z_max = action["z"]),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"], y_max = action["y"], z_min = action["z"] + 2, z_max = action["z"] + window_size),
			Region(x_min = action["x"], x_max = action["x"], y_min = action["y"], y_max = action["y"], z_min = action["z"] - window_size, z_max = action["z"] - 2)
		]
	else:
		rows_and_columns = []

	all_regions = [action_cell] + cells + rows_and_columns + quadrants + octants
	return all_regions

def get_block_counters(diffs_built_config_space, built_config, built_config_in_region, extra_check):
	"""
		Compute block counters for placements, next placements, removals and existing blocks in a region

		Args:
			built_config: Full built config -- used for feasibility checks for computing next placements
			built_config_in_region: Built config in the specific region -- used for computing existing blocks counter
	"""
	counters_per_diff = list(map(lambda x: region_to_counters(x, built_config, built_config_in_region, extra_check).__dict__, diffs_built_config_space))

	results = BlockCounters(None, None, None, None) # stores final result to return

	for field in ["all_placements_counter", "all_next_placements_counter", "all_removals_counter", "all_existing_blocks_counter"]:
		# obtain counters per diff per actions type
		counters_per_field = list(map(lambda x: x[field], counters_per_diff))

		# aggregate all and take expectations
		expectation_counter = sum(counters_per_field, Counter())
		for key in expectation_counter:
			expectation_counter[key] /= len(counters_per_field)

		# populate result obj
		if field == "all_placements_counter":
			results.all_placements_counter = expectation_counter
		elif field == "all_next_placements_counter":
			results.all_next_placements_counter = expectation_counter
		elif field == "all_removals_counter":
			results.all_removals_counter = expectation_counter
		elif field == "all_existing_blocks_counter":
			results.all_existing_blocks_counter = expectation_counter

	# reformat result obj
	def reformat(counter):
		all_colors_counter = [[]]

		for color in sorted(type2id.keys()):
			all_colors_counter[0].append(float(counter[color]))

		return all_colors_counter

	results.all_placements_counter = reformat(results.all_placements_counter)
	results.all_next_placements_counter = reformat(results.all_next_placements_counter)
	results.all_removals_counter = reformat(results.all_removals_counter)
	results.all_existing_blocks_counter = reformat(results.all_existing_blocks_counter)

	return results

def region_to_counters(a_diff, built_config, built_config_in_region, extra_check):
	"""
		Computer block counters for placements, next placements and removals for an optimal alignment
	"""
	def f(actions_list):
		actions_list_colors = list(map(lambda x: x["type"], actions_list))
		return Counter(actions_list_colors)

	# obtain all actions
	all_placements = a_diff["gold_minus_built"]
	all_next_placements = list(filter(lambda x: is_feasible_next_placement(x, built_config, extra_check), all_placements))
	all_removals = a_diff["built_minus_gold"]

	# map each set of actions to counters
	counts_all_placements = f(all_placements)
	counts_all_next_placements = f(all_next_placements)
	counts_all_removals = f(all_removals)

	# do same for existing blocks in region
	counts_all_existing_blocks = f(built_config_in_region)

	return BlockCounters(counts_all_placements, counts_all_next_placements, counts_all_removals, counts_all_existing_blocks)

class BlockCounters:
	"""
		Stores block counters for all action types
	"""
	def __init__(self, all_placements_counter, all_next_placements_counter, all_removals_counter, all_existing_blocks_counter):
		self.all_placements_counter = all_placements_counter
		self.all_next_placements_counter = all_next_placements_counter
		self.all_removals_counter = all_removals_counter
		self.all_existing_blocks_counter = all_existing_blocks_counter

def reformat_type_distributions(type_distributions_built_config_space):
	"""
	Args:
		type_distributions_built_config_space: Type distributions in built config space in the raw format
	Returns:
		a 4-d numpy array representation of the same with dimensions in the order type, x, y, z
	"""
	type_distributions_arr_built_config_space = np.zeros((len(type2id)+1, x_range, y_range, z_range))

	for elem in type_distributions_built_config_space:
		x = elem.grid_location["x"] - x_min
		y = elem.grid_location["y"] - y_min
		z = elem.grid_location["z"] - z_min

		for type in elem.type_distribution:
			type_id = len(type2id) if type == "empty" else type2id[type]
			probability = elem.type_distribution[type]
			type_distributions_arr_built_config_space[type_id][x][y][z] = probability

	return type_distributions_arr_built_config_space

def remove_empty_states(observations):
	observations["WorldStates"] = list(filter(lambda x: x["BuilderPosition"] != None, observations["WorldStates"]))
	return observations

def reorder(observations):
	"""
	Returns the observations dict by reordering blocks temporally in every state
	"""
	for i, state in enumerate(observations["WorldStates"]):
		prev_blocks = [] if i == 0 else observations["WorldStates"][i-1]["BlocksInGrid"]
		# pp.PrettyPrinter(indent=4).pprint(state)
		curr_blocks = state["BlocksInGrid"]
		curr_blocks_reordered = reorder_blocks(curr_blocks, prev_blocks) # obtain temporal ordering of blocks
		observations["WorldStates"][i]["BlocksInGrid"] = curr_blocks_reordered # mutate - will be used in next iteration

	return observations

def reorder_blocks(curr_blocks, prev_blocks):
	"""
	Returns a sorted version of the list of current blocks based on their order in the list of previous blocks.
	The assumption is that previous blocks are already sorted temporally.
	So this preserves that order for those blocks and puts any newly placed ones at the very end.
	"""
	return sorted(curr_blocks, key = lambda x: index(x, prev_blocks))

def index(curr_block, prev_blocks):
	"""
	Returns position of current block in the list of previous blocks.
	If not found in the list, returns a very large number (meaning it's a newly placed block and should be placed at the end when sorting temporally).
	"""
	for i, prev_block in enumerate(prev_blocks):
		if are_equal(curr_block, prev_block):
			return i

	return 999

def are_equal(block_1, block_2):
	"""
	Returns a comparison result between 2 blocks by ignoring the ever changing perspective coordinates
	"""
	return reformat(block_1) == reformat(block_2)

def get_last_action(curr_blocks, prev_blocks):
	curr_blocks = list(map(reformat, curr_blocks))
	prev_blocks = list(map(reformat, prev_blocks))

	diff_dict = diff(gold_config = curr_blocks, built_config = prev_blocks)
	all_actions = diff_dict["gold_minus_built"] + diff_dict["built_minus_gold"]

	return all_actions[0] if all_actions else None

def get_gold_actions(world_states):
	gold_placements, gold_removals = [], []
	next_world_state = None

	for i, world_state in reversed(list(enumerate(world_states))):
		# print(i, world_state["BlocksInGrid"])
		if not next_world_state:
			gold_placements.append([])
			gold_removals.append([])

		else:
			next_blocks = list(map(reformat, next_world_state['BlocksInGrid']))
			curr_blocks = list(map(reformat, world_state['BlocksInGrid']))
			diff_dict = diff(gold_config=next_blocks, built_config=curr_blocks)

			diff_dict['gold_minus_built'].extend(gold_placements[0])
			diff_dict['built_minus_gold'].extend(gold_removals[0])

			curr_blocks = set(map(dict_to_tuple, curr_blocks))
			removed_blocks = list(map(dict_to_tuple, diff_dict['built_minus_gold']))

			removed_existing = []
			for i2 in range(len(diff_dict['built_minus_gold'])):
				if removed_blocks[i2] in curr_blocks:
					removed_existing.append(diff_dict['built_minus_gold'][i2])

			gold_placements.insert(0, diff_dict['gold_minus_built'])
			gold_removals.insert(0, removed_existing)

		next_world_state = world_state

	return gold_placements, gold_removals

def format_prev_utterances(prev_utterances):
	for token in prev_utterances:
		print(self.encoder_vocab.idx2word[token], end=' ')
	print('\n')

if __name__ == '__main__':
	"""
	Use this section to generate datasets and for debugging purposes.
	BE CAREFUL TO NOT OVERWRITE EXISTING DATASETS AS DATASETS ARE NOT VERSION CONTROLLED.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='utterances_and_block_counters', help='model type')
	parser.add_argument('--split', default='train', help='dataset split')

	parser.add_argument('--dump_dataset', default=False, action='store_true', help='build the dataset')
	parser.add_argument('--lower', default=False, action='store_true', help='lowercase the dataset')
	parser.add_argument('--add_builder_utterances', default=False, action='store_true', help='add builder utterances')
	parser.add_argument('--add_augmented_data', default=False, action='store_true', help='add dialog-level augmented dataset')
	parser.add_argument('--ignore_diff', default=False, action='store_true', help='skip computing diff')
	parser.add_argument('--ignore_perspective', default=False, action='store_true', help='skip computing perspective coordinates')

	parser.add_argument('--load_dataset', default=False, action='store_true', help='load a dataset')
	parser.add_argument('--augmented_data_fraction', type=float, default=0.0, help='fraction of augmented data to use')
	parser.add_argument('--saved_dataset_dir', default="../data/saved_cwc_datasets/lower-no_perspective_coords/", help='location of saved dataset')
	parser.add_argument('--num_prev_utterances', type=int, default=5, help='number of previous utterances to use as input')
	parser.add_argument('--blocks_max_weight', type=int, default=5, help='max weight of temporally weighted blocks')
	parser.add_argument('--use_builder_actions', default=False, action='store_true', help='include builder action tokens in the dialogue history')
	parser.add_argument('--feasible_next_placements', default=False, action='store_true', help='whether or not to select from pool of feasible next placements only')
	parser.add_argument('--num_next_actions', type=int, default=2, help='number of next actions needed')
	parser.add_argument('--use_condensed_action_repr', default=False, action='store_true', help='use condensed action representation instead of one-hot')
	parser.add_argument('--action_type_sensitive', default=False, action='store_true', help='use action-type-sensitive representations for blocks')
	parser.add_argument('--spatial_info_window_size', type=int, default=1000, help='3d window size to extract spatial information from')
	parser.add_argument('--use_existing_blocks_counter', default=False, action='store_true', help='include counters for existing blocks')
	parser.add_argument('--counters_extra_feasibility_check', default=False, action='store_true', help='whether or not to make the extra check for conficting blocks')
	parser.add_argument('--encoder_vocab', default='../vocabulary/glove.42B.300d-lower-1r-speaker-builder_actions-oov_as_unk-all_splits/vocab.pkl', help='encoder vocab')
	parser.add_argument('--decoder_vocab', default='../vocabulary/glove.42B.300d-lower-2r-speaker-train_split-architect_only/vocab.pkl', help='decoder vocab')

	parser.add_argument('--seed', type=int, default=1234, help='random seed')

	args = parser.parse_args()

	initialize_rngs(args.seed, torch.cuda.is_available())

	if args.use_builder_actions and 'builder_actions' not in args.encoder_vocab:
		print("Error: you specified to use builder action tokens in the dialogue history, but they do not exist in the encoder's vocabulary.")
		sys.exit(0)

	with open(args.decoder_vocab, 'rb') as f:
		decoder_vocab = pickle.load(f)

	with open(args.encoder_vocab, 'rb') as f:
		encoder_vocab = pickle.load(f)

	# pr = cProfile.Profile()
	# pr.enable()

	dataset = CwCDataset(
		model=args.model, split=args.split, lower=args.lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff, compute_perspective=not args.ignore_perspective,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, dump_dataset=args.dump_dataset, load_dataset=args.load_dataset,
		saved_dataset_dir=args.saved_dataset_dir, transform=None, sample_filters = [], add_augmented_data=args.add_augmented_data, augmented_data_fraction=args.augmented_data_fraction
	)

	dataset.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, num_next_actions=args.num_next_actions, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)
	dl = dataset.get_data_loader(shuffle=False)

	for i in range(10):
		print('json id:', dataset.get_sample(i)['json_id'])
		print('sample id:', dataset.get_sample(i)['sample_id'])

		js = dataset.jsons[dataset.get_sample(i)['json_id']]
		print(js['gold_config_name'])
		print(js['WorldStates'][dataset.get_sample(i)['sample_id']])

	# pr.disable()
	# s = io.StringIO()
	# ps = pstats.Stats(pr, stream=s)
	# ps.print_stats()
	# print(s.getvalue())
	# pp.pprint(dataset.samples[10])

	# dpxs_placement, dpys_placement, dpzs_placement = [], [], []
	# dpxs_removal, dpys_removal, dpzs_removal = [], [], []
	# placement_null, removal_null, total = 0, 0, 0
	#
	# for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(dl):
	# 	next_actions = encoder_inputs.gold_actions
	# 	gold_next_placement = next_actions["gold_placements"]
	# 	if gold_next_placement[0][0][0].item() == 999:
	# 		placement_null += 1
	# 	else:
	# 		dpxs_placement.append(gold_next_placement[0][0][0].item())
	# 		dpys_placement.append(gold_next_placement[0][0][1].item())
	# 		dpzs_placement.append(gold_next_placement[0][0][2].item())
	#
	# 	gold_next_removal = next_actions["gold_removals"]
	# 	if gold_next_removal[0][0][0].item() == 999:
	# 		removal_null += 1
	# 	else:
	# 		dpxs_removal.append(gold_next_removal[0][0][0].item())
	# 		dpys_removal.append(gold_next_removal[0][0][1].item())
	# 		dpzs_removal.append(gold_next_removal[0][0][2].item())
	#
	# 	total += 1.0
	#
	# print("Null placements:", placement_null, "of", total, "total ("+str(placement_null/total*100)+'%)')
	# print(20*'-')
	# print("Placement: dx")
	# plot_histogram(dpxs_placement, "gold_next_placements_dx.png")
	# print(20*'-')
	#
	# print("Placement: dy")
	# plot_histogram(dpys_placement, "gold_next_placements_dy.png")
	# print(20*'-')
	#
	# print("Placement: dz")
	# plot_histogram(dpzs_placement, "gold_next_placements_dz.png")
	# print(20*'-')
	#
	# print("Null removals:", removal_null, "of", total, "total ("+str(removal_null/total*100)+'%)')
	# print(20*'-')
	# print("Removal: dx")
	# plot_histogram(dpxs_removal, "gold_next_removals_dx.png")
	# print(20*'-')
	#
	# print("Removal: dy")
	# plot_histogram(dpys_removal, "gold_next_removals_dy.png")
	# print(20*'-')
	#
	# print("Removal: dz")
	# plot_histogram(dpzs_removal, "gold_next_removals_dz.png")


		# if i == 10000: # 1622, 675 intresting examples for block counter info
		# 	break
		# pp.pprint(encoder_inputs.__dict__)

	for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(dl):
		print(raw_inputs.json_id, raw_inputs.sample_id)
		if i == 10:
			sys.exit(0)
		# pass
