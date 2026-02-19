#!/usr/bin/env bash

for seed in 0 1 2 3 4; do
    for dataset in dataset_RA_n479_n400 dataset_RA_n479_n400_n200 dataset_RA_n479_n400_n100; do

        mkdir -p storage/rad/${dataset}
        mkdir -p storage/instruct/${dataset}
        mkdir -p storage/combine/${dataset}

        uv run train.py --seed $seed --dataset datasets/${dataset}.pkl --save-dir storage/rad/${dataset} --log storage/rad/${dataset}/log_${seed}.csv
        uv run train.py --seed $seed --dataset datasets/${dataset}.pkl --save-dir storage/instruct/${dataset} --log storage/instruct/${dataset}/log_${seed}.csv --instruct
        uv run train.py --seed $seed --dataset datasets/${dataset}.pkl --save-dir storage/combine/${dataset} --log storage/combine/${dataset}/log_${seed}.csv --combine
    
    done
done
