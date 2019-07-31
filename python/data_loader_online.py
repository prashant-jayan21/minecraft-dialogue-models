import json

from data_loader import *
from utils import *
from vocab import Vocabulary

class CwCOnlineData(CwCDataset):

	def __init__(
		self, model, split, lower=False, add_builder_utterances=False, compute_diff=True, compute_perspective=True,
		augment_dataset=False, augmentation_factor=0, exactly_k=False, strict=False,
		data_dir="../data/logs/", gold_configs_dir="../data/gold-configurations/", save_dest_dir="../data/saved_cwc_datasets", saved_dataset_dir="../data/saved_cwc_datasets/lower-no_diff/", vocab_dir="../vocabulary/",
		encoder_vocab=None, decoder_vocab=None, dump_dataset=False, load_dataset=False, transform=None, sample_filters = [],
		add_augmented_data=False, augmented_data_fraction=0.0, aug_data_dir="../data/augmented-no-spatial/logs/", aug_gold_configs_dir="../data/augmented-no-spatial/gold-configurations/"
	):
		# NOTE: dump_dataset/load_dataset are ONLY TO BE USED WHEN YOU HAVE THE DIFF TO BE COMPUTED
		"""
		Args:
			split (string): which of train/test/dev split to be used. If none, then reads and stores all data.
			encoder_vocab (Vocabulary): encoder vocabulary wrapper.
			decoder_vocab (Vocabulary): decoder vocabulary wrapper.
			lower (boolean, optional): whether the data should be lowercased.
			data_dir (string): path to CwC official data directory.
			transform (callable, optional): transform to be applied on a sample.
		"""

		self.gold_configs_dir = gold_configs_dir

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

		self.online_data = True  # whether this if for architect demo or not aka online mode or not

	def set_args(self, num_prev_utterances=1, blocks_max_weight=1, use_builder_actions=False, num_next_actions=2, include_empty_channel=False, use_condensed_action_repr=False, action_type_sensitive=False, feasible_next_placements=False, spatial_info_window_size=1000, counters_extra_feasibility_check=False, use_existing_blocks_counter=False):
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

	def get_logfile_with_gold_config(self, config_structure, config_name, loaded_json):
	    loaded_json["gold_config_name"] = config_name
	    loaded_json["gold_config_structure"] = config_structure

	    return loaded_json

	def process_json(self, loaded_json):
	    loaded_json = remove_empty_states(reorder(loaded_json))

	    return loaded_json

	def process_sample(self, js, lower, compute_diff=True, compute_perspective=True):
	    """
	        only consider last world state for sample
	        from aug data is false
	    """
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
	                    chat_with_actions_history.append({"action": "putdown", "type": block["type"], "built_config": built_config, "prev_config": None, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

	                chat_with_actions_history.append({"action": "chat", "utterance": observation["ChatHistory"][i2].strip(), "built_config": built_config, "prev_config": None, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

	        else:
	            prev_config = get_built_config(last_world_state)
	            config_diff = diff(gold_config=built_config, built_config=prev_config)
	            delta = {"putdown": config_diff["gold_minus_built"], "pickup": config_diff["built_minus_gold"]}

	            for action_type in delta:
	                for block in delta[action_type]:
	                    chat_with_actions_history.append({"action": action_type, "type": block["type"], "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

	            if len(observation["ChatHistory"]) > len(last_world_state["ChatHistory"]):
	                for i3 in range(len(last_world_state["ChatHistory"]), len(observation["ChatHistory"])):
	                    chat_history.append(observation["ChatHistory"][i3].strip())
	                    chat_with_actions_history.append({"action": "chat", "utterance": observation["ChatHistory"][i3].strip(), "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "last_action": last_action, "gold_placement_list": gold_placement_list, "gold_removal_list": gold_removal_list})

	        last_world_state = observation

	    # process dialogue line-by-line
	    samples = []
	    for i in range(len(chat_with_actions_history)):
	        if i != len(chat_with_actions_history) - 1: # only consider last world state for sample
	            continue

	        elem = chat_with_actions_history[i]

	        # if elem['action'] != 'chat':
	        #     continue

	        line = elem['utterance']
	        built_config = elem["built_config"]
	        prev_config = elem["prev_config"]
	        builder_position = elem["builder_position"]
	        last_action = append_block_perspective_coords(builder_position, elem["last_action"])
	        gold_placement_list = [append_block_perspective_coords(builder_position, block) for block in elem["gold_placement_list"]]
	        gold_removal_list = [append_block_perspective_coords(builder_position, block) for block in elem["gold_removal_list"]]

	        speaker = "Architect" if "Architect" in line.split()[0] else "Builder"
	        # if not self.add_builder_utterances and speaker == 'Builder':
	        #     continue

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

	        # print(prev_utterances)
	        # print([prev_utterances[-1]])
	        samples.append(
	            {
	                'next_speaker': speaker,
	                'next_utterance': next_tokenized,
	                'prev_utterances': prev_utterances,
	                'gold_config': gold_config,
	                'built_config': built_config,
	                'diff': gold_v_built_diff,
	                'last_action': last_action,
	                'gold_placement_list': gold_placement_list,
	                'gold_removal_list': gold_removal_list,
	                'builder_position': builder_position,
	                'perspective_coordinates': perspective_coordinates,
	                'type_distributions_built_config_space': type_distributions_built_config_space,
	                'type_distributions_gold_config_space': type_distributions_gold_config_space,
	                'from_aug_data': False,
	                'diffs_built_config_space': diffs_built_config_space
	            }
	        )

	    # import pprint
	    # print(len(samples))
	    # pprint.PrettyPrinter(indent = 4).pprint(samples[0]["prev_utterances"])
	    # pprint.PrettyPrinter(indent = 4).pprint(samples[0]["next_utterance"])
	    self.samples = samples

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='utterances_and_next_actions', help='model type')
	parser.add_argument('--split', default='train', help='dataset split')

	parser.add_argument('--dump_dataset', default=False, action='store_true', help='build the dataset')
	parser.add_argument('--lower', default=False, action='store_true', help='lowercase the dataset')
	parser.add_argument('--add_builder_utterances', default=False, action='store_true', help='add builder utterances')
	parser.add_argument('--add_augmented_data', default=False, action='store_true', help='add dialog-level augmented dataset')
	parser.add_argument('--ignore_diff', default=False, action='store_true', help='skip computing diff')
	parser.add_argument('--ignore_perspective', default=False, action='store_true', help='skip computing perspective coordinates')

	parser.add_argument('--load_dataset', default=False, action='store_true', help='load a dataset')
	parser.add_argument('--augmented_data_fraction', type=float, default=0.0, help='fraction of augmented data to use')
	parser.add_argument('--saved_dataset_dir', default="../data/saved_cwc_datasets/lower/", help='location of saved dataset')
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

	dataset = CwCOnlineData(
		model=args.model, split=args.split, lower=args.lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff, compute_perspective=not args.ignore_perspective,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, dump_dataset=args.dump_dataset, load_dataset=args.load_dataset,
		saved_dataset_dir=args.saved_dataset_dir, transform=None, sample_filters = [], add_augmented_data=args.add_augmented_data, augmented_data_fraction=args.augmented_data_fraction
	)

	dataset.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, num_next_actions=args.num_next_actions, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)

	with open("../data/logs/data-3-30/logs/B1-A3-C1-1522435497386/postprocessed-observations.json") as json_data:
		loaded_json = json.load(json_data)

	dataset.process_sample(
		dataset.process_json(
			dataset.get_logfile_with_gold_config(
				gold_configs_dir = "../data/gold-configurations/", config_name = "C1", loaded_json = loaded_json
			)
		),
		lower=dataset.lower,
		compute_diff=dataset.compute_diff,
		compute_perspective=dataset.compute_perspective
	)

	dl = dataset.get_data_loader(shuffle=False)

	for i, (encoder_inputs, decoder_inputs, decoder_outputs, raw_inputs) in enumerate(dl):
		print(i)
		# import pprint
		# pprint.PrettyPrinter(indent=4).pprint(decoder_outputs.__dict__)
