import sys, torch, json, copy, pickle, re, os, numpy as np, pprint as pp, cProfile, pstats, io, traceback, itertools
sys.path.append('../../cwc-minecraft/build/install/Python_Examples/config_diff_tool')
from diff import diff, get_diff, get_next_actions, build_region_specs, dict_to_tuple, is_feasible_next_placement
from diff_apps import get_type_distributions

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter

from utils import *
from vocab import Vocabulary
from dataset_filters import *
from plot_utils import plot_histogram
from data_loader import CwCDataset

if __name__ == '__main__':
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

	saved_dataset_dir_old = "../data/saved_cwc_datasets/lower-no_perspective_coords-old_with_ids/"
	saved_dataset_dir_new = "../data/saved_cwc_datasets/lower-no_perspective_coords-fixed/"

	dataset_old = CwCDataset(
		model=args.model, split=args.split, lower=args.lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff, compute_perspective=not args.ignore_perspective,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, dump_dataset=args.dump_dataset, load_dataset=args.load_dataset,
		saved_dataset_dir=saved_dataset_dir_old, transform=None, sample_filters = [], add_augmented_data=args.add_augmented_data, augmented_data_fraction=args.augmented_data_fraction
	)

	dataset_old.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, num_next_actions=args.num_next_actions, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)
	dl_old = dataset_old.get_data_loader(shuffle=False)

	dataset_new = CwCDataset(
		model=args.model, split=args.split, lower=args.lower, add_builder_utterances=args.add_builder_utterances, compute_diff=not args.ignore_diff, compute_perspective=not args.ignore_perspective,
		encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, dump_dataset=args.dump_dataset, load_dataset=args.load_dataset,
		saved_dataset_dir=saved_dataset_dir_new, transform=None, sample_filters = [], add_augmented_data=args.add_augmented_data, augmented_data_fraction=args.augmented_data_fraction
	)

	dataset_new.set_args(num_prev_utterances=args.num_prev_utterances, blocks_max_weight=args.blocks_max_weight, use_builder_actions=args.use_builder_actions, num_next_actions=args.num_next_actions, use_condensed_action_repr=args.use_condensed_action_repr, action_type_sensitive=args.action_type_sensitive, feasible_next_placements=args.feasible_next_placements, spatial_info_window_size=args.spatial_info_window_size, counters_extra_feasibility_check=args.counters_extra_feasibility_check, use_existing_blocks_counter=args.use_existing_blocks_counter)
	dl_new = dataset_new.get_data_loader(shuffle=False)

	def compare_dialog_samples(old_dialog_samples, new_dialog_samples):
		def f(diff):
			built_minus_gold_reformatted = list(map(dict_to_tuple, diff["built_minus_gold"]))
			gold_minus_built_reformatted = list(map(dict_to_tuple, diff["gold_minus_built"]))

			return {
				"built_minus_gold": set(built_minus_gold_reformatted),
				"gold_minus_built": set(gold_minus_built_reformatted)
			}

		def g(diffs_1, diffs_2):
			diffs_1 = list(map(f, diffs_1))
			diffs_2 = list(map(f, diffs_2))

			return diffs_1 == diffs_2

		sample_diff_count = 0
		item_diff_count = 0
		all_diff_item_keys = []
		all_diff_sample_keys = []
		diff_samples = []
		diff_items = []
		for i, (old_sample, new_sample) in enumerate(zip(old_dialog_samples, new_dialog_samples)):
			same_sample = True
			diff_keys = []
			for k, v_old in old_sample.items():
				v_new = new_sample[k]
				if k in ["type_distributions_built_config_space", "type_distributions_gold_config_space"]:
					if np.array_equal(v_old, v_new):
						# print("SAME")
						pass
					else:
						# print(i)
						# print(k)
						# print("\n")
						# print("DIFF")
						same_sample = False
						diff_keys.append(k)
				elif k == "diff":
					if f(v_old) == f(v_new):
						# print("SAME")
						pass
					else:
						# print(i)
						# print(k)
						# print("\n")
						# print("DIFF")
						same_sample = False
						diff_keys.append(k)
				elif k == "diffs_built_config_space":
					if g(v_old, v_new):
						# print("SAME")
						pass
					else:
						# print(i)
						# print(k)
						# print("\n")
						# print("DIFF")
						same_sample = False
						diff_keys.append(k)
				elif k == "last_action":
					if v_old is not None and v_new is not None:
						v_old_reformatted = {
							"type": v_old["type"],
							"x": v_old["x"],
							"y": v_old["y"],
							"z": v_old["z"]
						}
						v_new_reformatted = {
							"type": v_new["type"],
							"x": v_new["x"],
							"y": v_new["y"],
							"z": v_new["z"]
						}
						if v_old_reformatted == v_new_reformatted:
							# print("SAME")
							pass
						else:
							# print(i)
							# print(k)
							# print("\n")
							# print("DIFF")
							same_sample = False
							diff_keys.append(k)
					else:
						if v_old == v_new:
							# print("SAME")
							pass
						else:
							# print(i)
							# print(k)
							# print("\n")
							# print("DIFF")
							same_sample = False
							diff_keys.append(k)
				elif k in ["gold_placement_list", "gold_removal_list", "sample_id"]:
					pass
				else:
					if v_old == v_new:
						# print("SAME")
						pass
					else:
						# print(i)
						# print(k)
						# if k in ["builder_position"]:
						# 	pp.PrettyPrinter(indent=4).pprint(v_old)
						# 	pp.PrettyPrinter(indent=4).pprint(v_new)
						# print(old_sample["next_utterance"])
						# print(new_sample["next_utterance"])
						# print("\n")
						# print("DIFF")
						same_sample = False
						diff_keys.append(k)
				# print("\n")

			if not same_sample:
				diff_samples.append(
					{
						"old_sample_id": old_sample["sample_id"],
						"new_sample_id": new_sample["sample_id"],
						"diff_sample_keys": sorted(diff_keys)
					}
				)

				sample_diff_count += 1
				print("diff keys in sample (except sample id)")
				print(diff_keys)
				all_diff_sample_keys += diff_keys

				encoder_inputs_old, decoder_inputs_old, decoder_outputs_old, _ = dataset_old.collate_fn([dataset_old.__getitem__(old_sample)])
				encoder_inputs_new, decoder_inputs_new, decoder_outputs_new, _ = dataset_new.collate_fn([dataset_new.__getitem__(new_sample)])

				def compare_encoder_inputs(inputs1, inputs2):
					diff_item_keys = []
					same = True
					if not torch.equal(inputs1.prev_utterances, inputs2.prev_utterances):
						# print("DIFF")
						print("diff key in item:", "prev_utterances")
						same = False
						diff_item_keys.append("prev_utterances")

					if not torch.equal(inputs1.last_action_bits, inputs2.last_action_bits):
						# print("DIFF")
						print("diff key in item:", "last_action_bits")
						same = False
						diff_item_keys.append("last_action_bits")

					if not torch.equal(inputs1.block_counters["all_placements_counter"], inputs2.block_counters["all_placements_counter"]) or not torch.equal(inputs1.block_counters["all_next_placements_counter"], inputs2.block_counters["all_next_placements_counter"]) or not torch.equal(inputs1.block_counters["all_removals_counter"], inputs2.block_counters["all_removals_counter"]):
						# print("DIFF")
						print("diff key in item:", "block_counters")
						same = False
						diff_item_keys.append("block_counters")

					comp1 = []
					for j in inputs1.block_counters_spatial_tensors:
						for k in j:
							comp1.append(k)

					comp2 = []
					for j in inputs2.block_counters_spatial_tensors:
						for k in j:
							comp2.append(k)

					same_ctr = True
					for x, y in zip(comp1, comp2):
						if not torch.equal(x, y):
							same_ctr = False
							break

					if not same_ctr:
						# print("DIFF")
						print("diff key in item:", "block_counters_spatial_tensors")
						same = False
						diff_item_keys.append("block_counters_spatial_tensors")

					return same, diff_item_keys

				def compare_decoder_inputs(inputs1, inputs2):
					diff_item_keys = []
					same = True
					if not torch.equal(inputs1.target_inputs, inputs2.target_inputs):
						# print("DIFF")
						print("diff key in item:", "target_inputs")
						same = False
						diff_item_keys.append("target_inputs")

					return same, diff_item_keys

				def compare_decoder_outputs(outputs1, outputs2):
					diff_item_keys = []
					same = True
					if not torch.equal(outputs1.target_outputs, outputs2.target_outputs):
						# print("DIFF")
						print("diff key in item:", "target_outputs")
						same = False
						diff_item_keys.append("target_outputs")

					return same, diff_item_keys

				same1, diff_item_keys_1 = compare_encoder_inputs(encoder_inputs_old, encoder_inputs_new)
				same2, diff_item_keys_2 = compare_decoder_inputs(decoder_inputs_old, decoder_inputs_new)
				same3, diff_item_keys_3 = compare_decoder_outputs(decoder_outputs_old, decoder_outputs_new)

				if not same1 or not same2 or not same3:
					item_diff_count += 1
					diff_item_keys = diff_item_keys_1 + diff_item_keys_2 + diff_item_keys_3

					diff_items.append(
						{
							"old_sample_id": old_sample["sample_id"],
							"new_sample_id": new_sample["sample_id"],
							"diff_sample_keys": sorted(diff_keys),
							"diff_item_keys": sorted(diff_item_keys)
						}
					)
				else:
					diff_item_keys = []

				all_diff_item_keys += diff_item_keys
				print("\n")

		print(sample_diff_count)
		print(item_diff_count)
		all_diff_item_keys = set(all_diff_item_keys)
		all_diff_sample_keys = set(all_diff_sample_keys)
		print("all_diff_item_keys")
		print(all_diff_item_keys)
		print("all_diff_sample_keys")
		print(all_diff_sample_keys)

		return sample_diff_count, item_diff_count, all_diff_item_keys, all_diff_sample_keys, diff_samples, diff_items

		# for old_sample in old_dialog_samples:
		# 	print(old_sample["next_utterance"])

	def groupby_dialog(samples):
		groups = []
		uniquekeys = []
		samples = sorted(samples, key=lambda x: x["json_id"])
		for k, g in itertools.groupby(samples, key=lambda x: x["json_id"]):
			groups.append(list(g))
			uniquekeys.append(k)

		return groups, uniquekeys

	groups_old, unique_dialogs_old = groupby_dialog(dataset_old.samples)
	groups_new, unique_dialogs_new = groupby_dialog(dataset_new.samples)

	assert unique_dialogs_old == unique_dialogs_new

	unique_dialogs = unique_dialogs_old

	all_sample_diff_count = 0
	all_item_diff_count = 0
	all_diff_sample_keys = []
	all_diff_item_keys= []

	results = []
	unequal_length_dialogs = []
	processed_samples = 0
	for i, (old_dialog_samples, new_dialog_samples) in enumerate(zip(groups_old, groups_new)):
		if i % 50 == 0:
			sys.stdout = sys.__stdout__
			print("Comparing dialog #" + str(i))
			sys.stdout = open(os.devnull, 'w')
		# if i == 100:
		# 	break

		json_id = unique_dialogs[i]

		if not len(old_dialog_samples) == len(new_dialog_samples):
			print("json id:", json_id)
			print("DIFF #SAMPLES IN DIALOG!!")
			unequal_length_dialogs.append(json_id)
		else:
			print("json id:", json_id)
			print("num samples:", len(old_dialog_samples))
			processed_samples += len(old_dialog_samples)
			print("\n")
			sample_diff_count, item_diff_count, diff_item_keys, diff_sample_keys, diff_samples, diff_items = compare_dialog_samples(old_dialog_samples, new_dialog_samples)
			all_sample_diff_count += sample_diff_count
			all_item_diff_count += item_diff_count
			all_diff_sample_keys += sorted(list(diff_sample_keys))
			all_diff_item_keys += sorted(list(diff_item_keys))
			print("\n\n")
			results.append({
				"json_id": json_id,
				"logfile_path": dataset_old.jsons[json_id]["logfile_path"],
				"num_samples": len(old_dialog_samples),
				"sample_diff_count": sample_diff_count,
				"item_diff_count": item_diff_count,
				"diff_sample_keys": sorted(list(diff_sample_keys)),
				"diff_item_keys": sorted(list(diff_item_keys)),
				"diff_samples": diff_samples,
				"diff_items": diff_items
			})

	all_diff_sample_keys = set(all_diff_sample_keys)
	all_diff_item_keys = set(all_diff_item_keys)

	# write results to file
	results_to_write = {
		"results": results,
		"dataset_stats": {
			"num_jsons": len(unique_dialogs),
			"num_old_samples": len(dataset_old.samples),
			"num_new_samples": len(dataset_new.samples),
			"processed_jsons": len(unique_dialogs) - len(unequal_length_dialogs),
			"processed_samples": processed_samples,
			"sample_diff_count": all_sample_diff_count,
			"item_diff_count": all_item_diff_count,
			"all_diff_sample_keys": sorted(list(all_diff_sample_keys)),
			"all_diff_item_keys": sorted(list(all_diff_item_keys))
		},
		"unprocessed_jsons": unequal_length_dialogs
	}

	with open("out-" + args.split + ".json", "w") as file:
		json.dump(results_to_write, file)
