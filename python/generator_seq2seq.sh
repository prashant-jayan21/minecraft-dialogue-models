#!/bin/bash
models_dir=""
beam_size=10
gamma=0.8
model_iteration='best'
split='val'
dev_mode=''
disable_shuffle=''
saved_dataset_dir="../data/saved_cwc_datasets/lower/"

for i in "$@"
do
case $i in
    --models_dir=*)
    models_dir="${i#*=}"
    ;;
    --beam_size=*)
    beam_size="${i#*=}"
    ;;
    --gamma=*)
    gamma="${i#*=}"
    ;;
    --model_iteration=*)
    model_iteration="${i#*=}"
    ;;
    --split=*)
	split="${i#*=}"
	;;
	--development_mode)
	dev_mode='--development_mode'
	;;
	--disable_shuffle)
	disable_shuffle='--disable_shuffle'
	;;
    *)
    ;;
esac
done

if [ "$models_dir" = "" ]; then
	echo "Error: please provide a directory in which models that should be evaluated reside."
	exit 1
fi

models_dir="$(cd "$(dirname "$models_dir")"; pwd)/$(basename "$models_dir")"

arr=(${models_dir}/*/)

for f in "${arr[@]}"; do
   python3 generate_seq2seq.py ${f} --saved_dataset_dir ${saved_dataset_dir} --split ${split} --beam_size ${beam_size} --gamma ${gamma} --model_iteration ${model_iteration} ${dev_mode} ${disable_shuffle} &
done
