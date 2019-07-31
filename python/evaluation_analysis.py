import sys, os, json, argparse, random
from bleu import compute_bleu

def get_terms(file):
	with open(file, 'r') as f:
		return [x.split()[0] for x in f.readlines()]

def read_generated_sentences_json(file):
	with open(file, 'r') as f:
		lines = f.readlines()
		bleu = lines[4].strip()
		return json.loads('\n'.join(lines[12:])), bleu

def color_stem(token):
	if token in ['reds', 'yellows', 'blues', 'oranges', 'greens', 'purples']:
		return token[:-1]
	return token

def get_simplified(utterance, simple_terms):
	return [color_stem(x) for x in utterance.split() if x in simple_terms]

def get_action_agreements(gold_next_placement, gold_next_removal, simplified_utterance):
	gold_action_agreement, gold_placement_agreement, gold_removal_agreement = 0, 0, 0
	if gold_next_placement in simplified_utterance or gold_next_removal in simplified_utterance:
		gold_action_agreement += 1

		if gold_next_placement and gold_next_placement in simplified_utterance:
			gold_placement_agreement += 1

		if gold_next_removal and gold_next_removal in simplified_utterance:
			gold_removal_agreement += 1

	return {'actions': gold_action_agreement, 'placements': gold_placement_agreement, 'removals': gold_removal_agreement}

def get_counter_agreements(next_placements_counter, all_placements_counter, all_removals_counter, simplified_utterance):
	next_placement_agreement, all_placement_agreement, all_removal_agreement = 0, 0, 0

	for color in simplified_utterance:
		if next_placement_agreement < 1 and next_placements_counter[color] > 0:
			next_placement_agreement += 1

		if all_placement_agreement < 1 and all_placements_counter[color] > 0:
			all_placement_agreement += 1

		if all_removal_agreement < 1 and all_removals_counter[color] > 0:
			all_removal_agreement += 1

	return {'next_placements': next_placement_agreement, 'all_placements': all_placement_agreement, 'all_removals': all_removal_agreement}

def update_map(actions, agreements):
	for action in agreements:
		actions[action] += agreements[action]

def update_totals(source, totals, next_placements_counter, all_placements_counter, all_removals_counter, simplified_utterance):
	totals['next_placements'][source] += 1.0 if sum(next_placements_counter.values()) > 0 and len(simplified_utterance) > 0 else 0.0
	totals['all_placements'][source] += 1.0 if sum(all_placements_counter.values()) > 0 and len(simplified_utterance) > 0 else 0.0
	totals['all_removals'][source] += 1.0 if sum(all_removals_counter.values()) > 0 and len(simplified_utterance) > 0 else 0.0

def format_decimal(num):
	return str('{0:.3g}'.format(num))

def get_synonimized_utterance(utterance, substitutions_lexicon):
	all_tokens = utterance

	def get_subs(token):
		# map token to list of all possible substitutions
		token_substitutions = [token]
		if token in substitutions_lexicon:
			token_substitutions += substitutions_lexicon[token]
		return token_substitutions

	# map each token to a list of it's substitutions including itself
	substitutions_list = list(map(get_subs, all_tokens))

	# sample
	all_tokens_substituted = list(map(lambda x: random.choice(x), substitutions_list))

	return all_tokens_substituted

def run_analysis(generated_sentences, original_bleu, terms, substitutions_lexicon, with_simple_synonyms, with_utterance_synonyms, num_synonym_references, output_file):
	reference_corpora, translation_corpora = {}, {}
	for term_type, _ in terms:
		reference_corpora[term_type] = []
		translation_corpora[term_type] = []

	if with_utterance_synonyms:
		reference_corpora['synonym'] = []
		translation_corpora['synonym'] = []

	totals = {'actions': 0.0, 'placements': 0.0, 'removals': 0.0, 'next_placements': {'gold': 0.0, 'generated': 0.0}, 'all_placements': {'gold': 0.0, 'generated': 0.0}, 'all_removals': {'gold': 0.0, 'generated': 0.0}}

	gold_utterance_agreement_map = {'actions': 0, 'placements': 0, 'removals': 0, 'next_placements': 0.0, 'all_placements': 0.0, 'all_removals': 0.0}
	gen_utterance_agreement_map = {'actions': 0, 'placements': 0, 'removals': 0, 'next_placements': 0.0, 'all_placements': 0.0, 'all_removals': 0.0}

	for sentence in generated_sentences:
		gold_utterance = sentence['ground_truth_utterance']
		generated_utterance = sentence['generated_utterance'][0]

		gold_next_action = sentence.get('gold_next_actions_raw')
		gold_next_placement = None if not gold_next_action or len(gold_next_action['gold_minus_built']) == 0 else gold_next_action['gold_minus_built'][0]['type']
		gold_next_removal = None if not gold_next_action or len(gold_next_action['built_minus_gold']) == 0 else gold_next_action['built_minus_gold'][0]['type']

		block_counters = sentence.get('block_counters')
		next_placements_counter = block_counters['all_next_placements_counter'] if block_counters else None
		all_placements_counter = block_counters['all_placements_counter'] if block_counters else None
		all_removals_counter = block_counters['all_removals_counter'] if block_counters else None

		output_file.write('original gold:      '+gold_utterance+'\n')
		if with_utterance_synonyms:
			reference_corpora['synonym'].append([gold_utterance.split()])
			translation_corpora['synonym'].append(generated_utterance.split())
		
			for i in range(num_synonym_references):
				synonym_reference = get_synonimized_utterance(gold_utterance.split(), substitutions_lexicon)
				output_file.write('original w synonym: '+' '.join(synonym_reference)+'\n')
				reference_corpora['synonym'][-1].append(synonym_reference)

		output_file.write('original generated: '+generated_utterance+'\n\n')

		output_file.write('gold next placement: '+str(gold_next_placement)+'\n')
		output_file.write('gold next removal:   '+str(gold_next_removal)+'\n')
		output_file.write('next placements counter: '+json.dumps(next_placements_counter)+'\n')
		output_file.write('all placements counter:  '+json.dumps(all_placements_counter)+'\n')
		output_file.write('all removals counter:    '+json.dumps(all_removals_counter)+'\n')

		for term_type, term_list in terms:
			reference_corpus, translation_corpus = reference_corpora[term_type], translation_corpora[term_type]
 
			simplified_gold = get_simplified(gold_utterance, term_list)
			simplified_generated = get_simplified(generated_utterance, term_list)

			reference_corpus.append([simplified_gold])
			translation_corpus.append(simplified_generated)
			
			output_file.write('\n'+term_type+' gold      --> '+str(simplified_gold)+'\n')
			if with_simple_synonyms and term_type != 'colors' and len(simplified_gold) > 0:
				for i in range(num_synonym_references):
					synonym_reference = get_synonimized_utterance(simplified_gold, substitutions_lexicon)
					output_file.write(term_type+' synonym   --> '+str(synonym_reference)+'\n')
					reference_corpus[-1].append(synonym_reference)

			output_file.write(term_type+' generated --> '+str(simplified_generated)+'\n')

			if term_type == 'colors':
				if gold_next_action:
					gold_utterance_action_agreements = get_action_agreements(gold_next_placement, gold_next_removal, simplified_gold)
					gen_utterance_action_agreements = get_action_agreements(gold_next_placement, gold_next_removal, simplified_generated)

					update_map(gold_utterance_agreement_map, gold_utterance_action_agreements)
					update_map(gen_utterance_agreement_map, gen_utterance_action_agreements)

					output_file.write('\ngold utterance action agreements:      '+json.dumps(gold_utterance_action_agreements)+'\n')
					output_file.write('generated utterance action agreements: '+json.dumps(gen_utterance_action_agreements)+'\n')

				if block_counters:
					gold_utterance_counter_agreements = get_counter_agreements(next_placements_counter, all_placements_counter, all_removals_counter, simplified_gold)
					gen_utterance_counter_agreements = get_counter_agreements(next_placements_counter, all_placements_counter, all_removals_counter, simplified_generated)

					update_map(gold_utterance_agreement_map, gold_utterance_counter_agreements)
					update_map(gen_utterance_agreement_map, gen_utterance_counter_agreements)

					output_file.write('\ngold utterance counter agreements:      '+json.dumps(gold_utterance_counter_agreements)+'\n')
					output_file.write('generated utterance counter agreements: '+json.dumps(gen_utterance_counter_agreements)+'\n')

					if len(simplified_gold) > 0:
						if gold_utterance_counter_agreements['next_placements'] < 1:
							output_file.write('***NO GOLD NEXT PLACEMENT AGREEMENT***\n')
						if gold_utterance_counter_agreements['all_placements'] < 1:
							output_file.write('***NO GOLD ALL PLACEMENT AGREEMENT***\n')
						if gold_utterance_counter_agreements['all_removals'] < 1:
							output_file.write('***NO GOLD ALL REMOVAL AGREEMENT***\n')

			totals['actions'] += 1.0
			totals['placements'] += 1.0 if gold_next_placement else 0.0
			totals['removals'] += 1.0 if gold_next_removal else 0.0

			if block_counters:
				update_totals('gold', totals, next_placements_counter, all_placements_counter, all_removals_counter, simplified_gold)
				update_totals('generated', totals, next_placements_counter, all_placements_counter, all_removals_counter, simplified_generated)

		output_file.write('-'*60+'\n')

	eval_str = '\n\n'+'original bleu-1: '.ljust(26)+str(original_bleu)+'\n'
	modified_bleu_scores = {}
	for term_type in reference_corpora:
		modified_bleu = compute_bleu(reference_corpora[term_type], translation_corpora[term_type], max_order=1, smooth=False)[0]
		eval_str += (term_type+' bleu-1: ').ljust(26)+str(modified_bleu)+'\n'
		modified_bleu_scores[term_type] = modified_bleu

	maps = {'gold': gold_utterance_agreement_map, 'generated': gen_utterance_agreement_map}
	agreements = {}

	for source in ['gold', 'generated']:
		for action in ['next_placements', 'all_placements', 'all_removals']: # 'actions', 'placements', 'removals', 
			total = totals[action][source] if action in ['next_placements', 'all_placements', 'all_removals'] else totals[action]
			if total == 0:
				continue
			agree_percent = maps[source][action]/total*100
			eval_str += '\n'+source+' utterance agreement with '+(action+': ').ljust(17)+str(maps[source][action])+'/'+str(total)+' ('+format_decimal(agree_percent)+' %)'
			if source == 'generated':
				agreements[action] = agree_percent
		eval_str += '\n'

	output_file.write(eval_str)
	return modified_bleu_scores, agreements, eval_str

def main(args):
	random.seed(1234)
	generated_sentences, original_bleu = read_generated_sentences_json(args.generated_sentences_file)
	with open(args.synonyms_file, 'r') as f:
		substitutions_lexicon = json.loads(f.read())
	terms = [('simple_terms', get_terms(args.simple_terms_file)), ('colors', get_terms(args.colors_file)), ('spatial_relations', get_terms(args.spatial_relations_file)), ('dialogue', get_terms(args.dialogue_file))] 
	output_file = open(args.output_file, 'w') if args.output_file else open(os.path.join('/'.join(args.generated_sentences_file.split('/')[:-1]), 'modified-bleu-eval.txt'), 'w')
	modified_bleu_scores, agreements, eval_str = run_analysis(generated_sentences, original_bleu, terms, substitutions_lexicon, args.with_simple_synonyms, args.with_utterance_synonyms, args.num_synonym_references, output_file)
	if not args.suppress_printing:
		print(eval_str.strip())
	return modified_bleu_scores, agreements

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('generated_sentences_file', help='file of sentences generated by a model')
	parser.add_argument('--simple_terms_file', default='../data/lexicons/simple-terms-redux.txt')
	parser.add_argument('--colors_file', default='../data/lexicons/colors.txt')
	parser.add_argument('--spatial_relations_file', default='../data/lexicons/spatial-relations.txt')
	parser.add_argument('--dialogue_file', default='../data/lexicons/dialogue.txt')
	parser.add_argument('--shapes_file', default='../data/lexicons/shapes.txt')
	parser.add_argument('--synonyms_file', default='../data/lexicons/synonym_substitutions.json')
	parser.add_argument('--with_simple_synonyms', default=False, action='store_true')
	parser.add_argument('--with_utterance_synonyms', default=False, action='store_true')
	parser.add_argument('--num_synonym_references', default=4)
	parser.add_argument('--suppress_printing', default=False, action='store_true')
	parser.add_argument('--output_file', default=None)
	args = parser.parse_args()
	main(args)