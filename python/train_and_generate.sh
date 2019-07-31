#!/bin/bash
gpu_id=0
model_type="utterances_and_block_region_counters"
hyperparameters_file="../config/utterances_and_block_counters/hyperparameters-1.config"
beam_size=10
gamma=0.8
timestamp=$(date +%s)
criterion="xent"
date_dir=""

for i in "$@"
do
case $i in
    --gpu_id=*)
    gpu_id="${i#*=}"
    ;;
    --model_type=*)
    model_type="${i#*=}"
    ;;
    --hyperparameters_file=*)
    hyperparameters_file="${i#*=}"
    ;;
    --beam_size=*)
    beam_size="${i#*=}"
    ;;
    --gamma=*)
    gamma="${i#*=}"
    ;;
    --criterion=*)
    criterion="${i#*=}"
    ;;
    --date_dir=*)
    date_dir="--date_dir ${i#*=}"
    ;;
    *)
    ;;
esac
done

export CUDA_VISIBLE_DEVICES=${gpu_id}

trainer_script="trainer.py"
if [ ${criterion} == "mrl" ]; then
    trainer_script="trainer_mrl.py"
fi

if [ "$(uname)" == "Darwin" ]; then
    script -q /dev/null python3 ${trainer_script} ${model_type} ${hyperparameters_file} ${date_dir} | tee "temp-${timestamp}.txt"
else
    script -q -c "python3 ${trainer_script} ${model_type} ${hyperparameters_file} ${date_dir}" /dev/null | tee "temp-${timestamp}.txt"
fi

models_dir=$(tail -1 "temp-${timestamp}.txt")
rm "temp-${timestamp}.txt"
export CUDA_VISIBLE_DEVICES=-1
python3 generator_seq2seq.py ${models_dir//$'\r'/} --beam_size ${beam_size} --gamma ${gamma} --disable_shuffle