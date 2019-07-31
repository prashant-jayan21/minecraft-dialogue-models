import sys, os, re, json, argparse, random, nltk, torch, pickle, numpy as np, copy
from glob import glob
from datetime import datetime
from os.path import join, isdir
import xml.etree.ElementTree as ET
from torch.autograd import Variable
from sklearn.model_selection import train_test_split as tt_split

color_regex = re.compile("red|orange|purple|blue|green|yellow") # TODO: Obtain from other repo

# assigning IDs to block types aka colors
type2id = {
	"orange": 0,
	"red": 1,
	"green": 2,
	"blue": 3,
	"purple": 4,
	"yellow": 5
}

# assigning IDs to block placement/removal actions
action2id = {
	"placement": 0,
	"removal": 1
}

# bounds of the build region
x_min = -5
x_max = 5
y_min = 1
y_max = 9
z_min = -5
z_max = 5 # TODO: Obtain from other repo
x_range = x_max - x_min + 1
y_range = y_max - y_min + 1
z_range = z_max - z_min + 1

class EvaluationResult(object):
	"""
		Stores loss and perplexity
	"""
	def __init__(self, loss, num_words=0):
		self.eval_dict = {}
		self.eval_dict["Loss"] = loss # total loss on data set
		self.eval_dict["Perplexity"] = np.exp(loss / num_words) if num_words > 0 else float('-inf') # loss per word

	def __call__(self, key):
		if not key in self.eval_dict:
			return None
		return self.eval_dict[key]

	def __str__(self):
		eval_string = ""
		for key in self.eval_dict:
			if self.eval_dict[key] > float('-inf'):
				eval_string += key+": %5.4f | " %(self.eval_dict[key])
		return eval_string.strip()[:-1]

	def pretty_print(self, sep=', '):
		eval_string = ""
		for key in self.eval_dict:
			if self.eval_dict[key] > float('-inf'):
				eval_string += key+": %5.4f" %(self.eval_dict[key])+sep

		eval_string = eval_string.strip()
		if sep == ', ':
			eval_string = eval_string[:-1]

		return eval_string

class Logger(object):
	""" Simple logger that writes messages to both console and disk. """

	def __init__(self, logfile_path):
		"""
		Args:
			logfile_path (string): path to where the log file should be saved.
		"""
		self.terminal = sys.stdout
		self.log = open(logfile_path, "a")

	def write(self, message):
		""" Writes a message to both stdout and logfile. """
		self.terminal.write(message)
		self.log.write(message)
		self.log.flush()

	def flush(self):
		pass

class EncoderContext:
	"""
		Output of an encoder set up for use in a corresponding decoder
			- decoder_hidden, decoder_input_concat, etc. point to various ways of conditioning the decoder on the encoder's output
			- Each is initialized appropriately with the the encoder's output so as to be used in the decoder
	"""
	def __init__(self, decoder_hidden=None, decoder_input_concat=None, decoder_hidden_concat=None, decoder_input_t0=None, attn_vec=None):
		self.decoder_hidden = decoder_hidden
		self.decoder_input_concat = decoder_input_concat
		self.decoder_hidden_concat = decoder_hidden_concat
		self.decoder_input_t0 = decoder_input_t0
		self.attn_vec = attn_vec

def get_logfiles(data_path, split=None):
	"""
	Gets all CwC observation files along without the corresponding gold config. According to a given split.
	Split can be "train", "test" or "val"
	"""
	return get_logfiles_with_gold_config(data_path=data_path, gold_configs_dir=None, split=split, with_gold_config=False)

def get_logfiles_with_gold_config(data_path, gold_configs_dir, split=None, with_gold_config=True, from_aug_data=False):
	"""
	Gets all CwC observation files along with the corresponding gold config, according to a given split.
	Split can be "train", "test" or "val"
	"""

	# get required configs
	with open(data_path + "/splits.json") as json_data:
			data_splits = json.load(json_data)

	configs_for_split = data_splits[split]

	# get all postprocessed observation files along with gold config data
	jsons = []

	all_data_root_dirs = filter(lambda x: isdir(join(data_path, x)), os.listdir(data_path))
	for data_root_dir in all_data_root_dirs:
		logs_root_dir = join(data_path, data_root_dir, "logs")

		all_log_dirs = filter(lambda x: isdir(join(logs_root_dir, x)), os.listdir(logs_root_dir))
		for log_dir in all_log_dirs:
			config_name = re.sub(r"B\d+-A\d+-|-\d\d\d\d\d\d\d+", "", log_dir)

			if config_name not in configs_for_split:
				continue

			if with_gold_config:
				config_xml_file = join(gold_configs_dir, config_name + ".xml")
				config_structure = get_gold_config(config_xml_file)

			logfile = join(logs_root_dir, log_dir, "postprocessed-observations.json")
			with open(logfile) as f:
				loaded_json = json.loads(f.read())
				loaded_json["from_aug_data"] = from_aug_data

				if with_gold_config:
					loaded_json["gold_config_name"] = config_name
					loaded_json["gold_config_structure"] = config_structure
					loaded_json["log_dir"] = log_dir
					loaded_json["logfile_path"] = logfile

				jsons.append(loaded_json)

	return jsons

def get_gold_config(gold_config_xml_file): # TODO: Obtain from other repo
	"""
	Args:
		gold_config_xml_file: The XML file for a gold configuration

	Returns:
		The gold config as a list of dicts -- one dict per block
	"""
	with open(gold_config_xml_file) as f:
		all_lines = map(lambda t: t.strip(), f.readlines())

	gold_config_raw = map(ET.fromstring, all_lines)

	displacement = 100 # TODO: Obtain from other repo
	def reformat(block):
		return {
			"x": int(block.attrib["x"]) - displacement,
			"y": int(block.attrib["y"]),
			"z": int(block.attrib["z"]) - displacement,
			"type": color_regex.findall(block.attrib["type"])[0]
		}

	gold_config = list(map(reformat, gold_config_raw))

	return gold_config

def get_built_config(observation):
	"""
	Args:
		observation: The observations for a cetain world state

	Returns:
		The built config for that state as a list of dicts -- one dict per block
	"""

	built_config_raw = observation["BlocksInGrid"]
	built_config = list(map(reformat, built_config_raw))
	return built_config

def get_builder_position(observation):
	builder_position = observation["BuilderPosition"]

	builder_position = {
		"x": builder_position["X"],
		"y": builder_position["Y"],
		"z": builder_position["Z"],
		"yaw": builder_position["Yaw"],
		"pitch": builder_position["Pitch"]
	}

	return builder_position

def reformat(block):
	return {
		"x": block["AbsoluteCoordinates"]["X"],
		"y": block["AbsoluteCoordinates"]["Y"],
		"z": block["AbsoluteCoordinates"]["Z"],
		"type": color_regex.findall(str(block["Type"]))[0] # NOTE: DO NOT CHANGE! Unicode to str conversion needed downstream when stringifying the dict.
	}

def to_var(x, volatile=False):
	""" Returns an input as a torch Variable, cuda-enabled if available. """
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def timestamp():
	""" Simple timestamp marker for logging. """
	return "["+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"]"

def print_dir(path, n):
	path = os.path.abspath(path).split("/")
	return "/".join(path[len(path)-n:])

def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

def ids2words(vocabulary, sampled_ids):
	sampled_utterance = []
	for word_id in sampled_ids:
		word = vocabulary.idx2word[word_id]
		sampled_utterance.append(word)
	return ' '.join(sampled_utterance)

def tokenize(utterance):
	tokens = utterance.split()
	fixed = ""

	modified_tokens = set()
	for token in tokens:
		original = token

		# fix *word
		if len(token) > 1 and token[0] == '*':
			token = '* '+token[1:]

		# fix word*
		elif len(token) > 1 and token[-1] == '*' and token[-2] != '*':
			token = token[:-1]+' *'

		# fix word..
		if len(token) > 2 and token[-3] is not '.' and ''.join(token[-2:]) == '..':
			token = token[:-2]+' ..'

		# split axb(xc) to a x b (x c)
		if len(token) > 2:
			m = re.match("([\s\S]*\d+)x(\d+[\s\S]*)", token)
			while m:
				token = m.groups()[0]+' x '+m.groups()[1]
				m = re.match("([\s\S]*\d+)x(\d+[\s\S]*)", token)

		if original != token:
			modified_tokens.add(original+' -> '+token)

		fixed += token+' '

	return nltk.tokenize.word_tokenize(fixed.strip()), modified_tokens

def get_config_params(config_file):
	with open(config_file, 'r') as f:
		config_content = f.read()

	config_params = {}
	ignore_params = ['model_path', 'data_dir', 'log_step', 'epochs', 'stop_after_n', 'num_workers', 'seed', 'suppress_logs']

	for line in config_content.split('\n'):
		if len(line.split()) != 2:
			continue
		(param, value) = line.split()
		if not any(ignore_param in param for ignore_param in ignore_params):
			config_params[param] = parse_value(value)

	return config_content, config_params

def parse_value(value):
	if value == 'None':
		return None

	try:
		return int(value)
	except ValueError:
		try:
			return float(value)
		except ValueError:
			if value.lower() == 'true' or value.lower() == 'false':
				return str2bool(value)
			return value

def load_pkl_data(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		print("Loaded data from '%s'" %os.path.realpath(f.name))

	return data

def save_pkl_data(filename, data, protocol=3):
	with open(filename, 'wb') as f:
		pickle.dump(data, f, protocol=protocol)
		print("Saved data to '%s'" %os.path.realpath(f.name))

def get_action_type_repr(action_type):
	action_type = action2id[action_type]
	action_type_one_hot_vec = [0] * len(action2id)
	action_type_one_hot_vec[action_type] = 1
	return action_type_one_hot_vec

def get_one_hot_repr(blocks_sequence, action_type_sensitive=False, action_type_null_case=None):

	def get_one_hot_repr_block(block, action_type_sensitive):
		x = int(block["x"])
		y = int(block["y"])
		z = int(block["z"])
		type = type2id[block["type"]]

		x_one_hot_vec = [0] * x_range
		if x >= x_min and x <= x_max:
			x_one_hot_vec[x - x_min] = 1

		y_one_hot_vec = [0] * y_range
		if y >= y_min and y <= y_max:
			y_one_hot_vec[y - y_min] = 1

		z_one_hot_vec = [0] * z_range
		if z >= z_min and z <= z_max:
			z_one_hot_vec[z - z_min] = 1

		type_one_hot_vec = [0] * len(type2id)
		type_one_hot_vec[type] = 1

		canonical_encoding = x_one_hot_vec + y_one_hot_vec + z_one_hot_vec + type_one_hot_vec

		if not action_type_sensitive:
			return canonical_encoding
		else:
			action_type_one_hot_vec = get_action_type_repr(block["action_type"])
			# action_type = action2id[block["action_type"]]
			# action_type_one_hot_vec = [0] * len(action2id)
			# action_type_one_hot_vec[action_type] = 1

			return canonical_encoding + action_type_one_hot_vec

	if blocks_sequence == []:
		canonical_encoding_size = x_range + y_range + z_range + len(type2id)

		if not action_type_sensitive:
			blocks_sequence_repr = [[0] * canonical_encoding_size]
		else:
			action_type_one_hot_vec = get_action_type_repr(action_type_null_case)
			# action_type = action2id[action_type_null_case]
			# action_type_one_hot_vec = [0] * len(action2id)
			# action_type_one_hot_vec[action_type] = 1

			blocks_sequence_repr = [[0] * canonical_encoding_size + action_type_one_hot_vec]
	else:
		blocks_sequence_repr = list(map(lambda x: get_one_hot_repr_block(x, action_type_sensitive), blocks_sequence))

	return blocks_sequence_repr

def get_condensed_repr(blocks_sequence, last_action, action_type_sensitive, action_type_null_case):
	if len(blocks_sequence) < 1:
		if action_type_sensitive:
			action_type_one_hot_vec = get_action_type_repr(action_type_null_case)
			return [[999]*3+[0]*len(type2id)+action_type_one_hot_vec]

		return [[999]*3+[0]*len(type2id)]

	def get_condensed_repr_block(block, last_action, action_type_sensitive):
		dpx = float(block['px'])-(0 if not last_action else float(last_action['px']))
		dpy = float(block['py'])-(0 if not last_action else float(last_action['py']))
		dpz = float(block['pz'])-(0 if not last_action else float(last_action['pz']))
		block_type = type2id[block["type"]]

		canonical_encoding = [dpx, dpy, dpz]
		type_one_hot_vec = [0] * len(type2id)
		type_one_hot_vec[block_type] = 1
		canonical_encoding.extend(type_one_hot_vec)

		if action_type_sensitive:
			action_type_one_hot_vec = get_action_type_repr(block["action_type"])
			canonical_encoding.extend(action_type_one_hot_vec)

		return canonical_encoding

	return list(map(lambda x: get_condensed_repr_block(x, last_action, action_type_sensitive), blocks_sequence))

def get_3d_repr(blocks_sequence, max_weight=5, include_empty_channel=False):
	num_channels = len(type2id)+1 if include_empty_channel else len(type2id)
	config_3d = torch.zeros(num_channels, x_range, y_range, z_range)
	all_blocks = set()
	weight = max_weight

	for block in reversed(blocks_sequence):
		x, y, z = block["x"]-x_min, block["y"]-y_min, block["z"]-z_min
		block_type = type2id[block["type"]]
		all_blocks.add((x,y,z))

		# fixme: ignoring outside blocks
		if x < 0 or x >= x_range or y < 0 or y >= y_range or z < 0 or z >= z_range:
			continue

		config_3d[block_type][x][y][z] = weight

		if weight > 1:
			weight -= 1

	if include_empty_channel:
		for x in range(x_range):
			for y in range(y_range):
				for z in range(z_range):
					if (x,y,z) not in all_blocks:
						config_3d[len(type2id)][x][y][z] = 1

	return config_3d

def get_perspective_coordinates(x, y, z, yaw, pitch):
	# construct vector
	v = np.matrix('{}; {}; {}'.format(x, y, z))

	# construct yaw rotation matrix
	theta_yaw = np.radians(-1 * yaw)
	c, s = np.cos(theta_yaw), np.sin(theta_yaw)
	R_yaw = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, 0, -s, 0, 1, 0, s, 0, c))

	# multiply
	v_new = R_yaw * v

	# construct pitch rotation matrix
	theta_pitch = np.radians(pitch)
	c, s = np.cos(theta_pitch), np.sin(theta_pitch)
	R_pitch = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0, 0, c, s, 0, -s, c))

	# multiply
	v_final = R_pitch * v_new
	x_final = v_final.item(0)
	y_final = v_final.item(1)
	z_final = v_final.item(2)
	return (x_final, y_final, z_final)

vf = np.vectorize(get_perspective_coordinates)

def append_block_perspective_coords(builder_position, block):
	if not block:
		return None

	block_copy = copy.deepcopy(block)

	bx = builder_position["x"]
	by = builder_position["y"]
	bz = builder_position["z"]
	yaw = builder_position["yaw"]
	pitch = builder_position["pitch"]

	x = block_copy["x"]
	y = block_copy["y"]
	z = block_copy["z"]

	xm, ym, zm = x-bx, y-by, z-bz
	px, py, pz = get_perspective_coordinates(xm, ym, zm, yaw, pitch)

	block_copy["px"] = px
	block_copy["py"] = py
	block_copy["pz"] = pz

	return block_copy

def get_perspective_coord_repr(builder_position):
	bx = builder_position["x"]
	by = builder_position["y"]
	bz = builder_position["z"]
	yaw = builder_position["yaw"]
	pitch = builder_position["pitch"]

	perspective_coords = np.zeros((3, x_range, y_range, z_range))
	for x in range(x_range):
		for y in range(y_range):
			for z in range(z_range):
				xm, ym, zm = x-bx, y-by, z-bz
				perspective_coords[0][x][y][z] = xm
				perspective_coords[1][x][y][z] = ym
				perspective_coords[2][x][y][z] = zm

	px, py, pz = vf(perspective_coords[0], perspective_coords[1], perspective_coords[2], yaw, pitch)
	return np.stack([px, py, pz])

def get_next_actions_repr(next_actions, last_action, action_type_sensitive, use_condensed_action_repr=False):
	# {
	#     "gold_minus_built": next_placements,
	#     "built_minus_gold": next_removals
	# }

	def add_action_type(action, placement_or_removal):
		assert placement_or_removal in ["placement", "removal"]

		action_copy = copy.deepcopy(action)
		action_copy["action_type"] = placement_or_removal

		return action_copy

	next_placements = next_actions["gold_minus_built"]
	next_placements = list(map(lambda x: add_action_type(x, "placement"), next_placements))
	next_removals = next_actions["built_minus_gold"]
	next_removals = list(map(lambda x: add_action_type(x, "removal"), next_removals))

	if not use_condensed_action_repr:
		next_placements_repr = get_one_hot_repr(next_placements, action_type_sensitive, action_type_null_case="placement")
		next_removals_repr = get_one_hot_repr(next_removals, action_type_sensitive, action_type_null_case="removal")
	else:
		next_placements_repr = get_condensed_repr(next_placements, last_action, action_type_sensitive, action_type_null_case='placement')
		next_removals_repr = get_condensed_repr(next_removals, last_action, action_type_sensitive, action_type_null_case='removal')

	return {
		"next_placements_repr": next_placements_repr,
		"next_removals_repr": next_removals_repr
	}

architect_prefix = "<Architect> "
builder_prefix = "<Builder> "

def get_data_splits(args):
	"""
	Writes a file containing the train-val-test splits at the config level
	"""

	# utils
	warmup_configs_blacklist = ["C3", "C17", "C32", "C38"] # TODO: import from another repo

	# get all gold configs

	gold_configs = []

	for gold_config_xml_file in glob(args.gold_configs_dir + '/*.xml'):
		gold_config = gold_config_xml_file.split("/")[-1][:-4]
		gold_configs.append(gold_config)

	# filter out warmup ones
	gold_configs = list(filter(lambda x: x not in warmup_configs_blacklist, gold_configs))

	# split
	train_test_split = tt_split(gold_configs, random_state=args.seed) # default is 0.75:0.25

	train_configs = train_test_split[0]
	test_configs = train_test_split[1]

	train_val_split = tt_split(train_configs, random_state=args.seed) # default is 0.75:0.25

	train_configs = train_val_split[0]
	val_configs = train_val_split[1]

	# write split to file
	splits = {
		"train": train_configs,
		"val": val_configs,
		"test": test_configs
	}

	with open(args.data_path + "/splits.json", "w") as file:
		json.dump(splits, file)

def get_num_colors():
	return len(type2id)

def str2bool(v):
	return v.lower() == "true"

def get_augmented_data_splits(data_path, gold_configs_dir, splits_json_for_orig_data):

	def find_set(orig_gold_config, orig_data_splits):
		if orig_gold_config in orig_data_splits["train"]:
			return "train"
		elif orig_gold_config in orig_data_splits["val"]:
			return "val"
		elif orig_gold_config in orig_data_splits["test"]:
			return "test"
		else:
			return None # warmup config

	# load original data splits
	with open(splits_json_for_orig_data) as json_data:
            orig_data_splits = json.load(json_data)

	# get all gold configs in augmented data
	gold_configs = []

	for gold_config_xml_file in glob(gold_configs_dir + '/*.xml'):
		gold_config = gold_config_xml_file.split("/")[-1][:-4]
		gold_configs.append(gold_config)

	# split

	aug_data_splits = {
		"train": [],
		"val": [],
		"test": []
	}

	for gold_config in gold_configs:
		# find right set -- train/test/val
		corresponding_orig_gold_config = gold_config.split("_")[0]
		split_set = find_set(corresponding_orig_gold_config, orig_data_splits)
		# assign to a set iff it's not a warmup config
		if split_set:
			aug_data_splits[split_set].append(gold_config)

	with open(data_path + "/splits.json", "w") as f:
		json.dump(aug_data_splits, f)

def initialize_rngs(seed, use_cuda=False):
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        # torch.backends.cudnn.deterministic = True  #needed
        # torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
	"""
	Use this section for generating the splits files (you shouldn't need to run this -- think carefully about what you are doing).
	"""

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_path', type=str, default='../data/logs/', help='path for data jsons')
	parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/', help='path for gold config xmls')

	parser.add_argument('--seed', type=int, default=1234, help='random seed')

	args = parser.parse_args()

	initialize_rngs(args.seed, torch.cuda.is_available())

	get_data_splits(args)

	get_augmented_data_splits(
		"../data/augmented-no-spatial/logs/", "../data/augmented-no-spatial/gold-configurations/", "../data/logs/splits.json"
	)
