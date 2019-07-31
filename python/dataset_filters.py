# DATASET FILTERS
# NOTE: DO NOT CHANGE THE NAME OR BEHAVIOR OF THESE FILTER FUNCTIONS -- DO NOT TOUCH BASICALLY!
# IF YOU WANT A DIFFERENT FILTER, JUST ADD A NEW ONE

def filter_1(sample):
	next_utterance = sample["next_utterance"] # ['he', 'is', 'a', 'man']
	colors = ["red", "orange", "purple", "blue", "green", "yellow"]
	colors_in_utterance = set(colors) & set(next_utterance)

	return len(colors_in_utterance) == 1

def filter_2(sample):
	next_utterance = sample["next_utterance"] # ['he', 'is', 'a', 'man']

	return len(next_utterance) <= 10
