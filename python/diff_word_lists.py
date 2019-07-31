import sys, os, argparse

def read_word_freqs(file):
	word_freqs = {}
	with open(file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			tokens = line.split()
			word, freq = tokens[0], int(tokens[1].strip())
			word_freqs[word] = freq

	return word_freqs

def main(args):
	reference_words = read_word_freqs(args.reference_words)
	diff_words = read_word_freqs(args.diff_words)
	values = { k : reference_words[k] for k in set(reference_words) - set(diff_words) }
	values = sorted(values.items(), key=lambda x: x[1], reverse=True)
	for word, freq in values:
		print(word.ljust(30), freq)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--reference_words', type=str, default='../data/lexicons/all-words.txt')
	parser.add_argument('--diff_words', type=str, default='../data/lexicons/spatial-relations.txt')
	args = parser.parse_args()
	main(args)
