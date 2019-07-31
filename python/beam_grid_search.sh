#!/bin/bash
model="../models/cnn_3d/20181023/1540369400463"
if [ "$1" != "" ]; then
	model=$1
fi

gamma=( 0.0 0.2 0.5 0.8 1.0)
beam=(5 10 20 30)

for g in "${gamma[@]}"
do
	for b in "${beam[@]}"
	do
		python3 generate_seq2seq.py ${model} --beam_size ${b} --gamma ${g} &
	done
done
