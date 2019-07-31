#!/bin/bash
models_dir=""
beam_size=10
gamma=0.8

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
    *)
    ;;
esac
done

export CUDA_VISIBLE_DEVICES=-1
python3 generator_seq2seq.py ${models_dir//$'\r'/} --beam_size ${beam_size} --gamma ${gamma} --disable_shuffle