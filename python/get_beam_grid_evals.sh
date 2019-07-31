#!/bin/bash
model_dir=$1
awk ' { print FILENAME "," $0 } ' ${model_dir}/*.csv > ${model_dir}/beam_grid_search_evals.csv
echo -e "filename, mean utterance length (val), std dev (val), bleu-1 (val), bleu-2 (val), bleu-3 (val), bleu-4 (val)\n$(cat ${model_dir}/beam_grid_search_evals.csv)" > ${model_dir}/beam_grid_search_evals.csv