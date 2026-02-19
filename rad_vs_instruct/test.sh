#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <N_samples> <csv_file>"
  exit 1
fi

N=$1
CSV=$2

echo "Type, Train Dataset, Test Dataset, Success Probability, Episode Length, Episode Reward, Episode Discounted Reward" > $CSV
# storage/rad/dataset_RA_n479_n400_n100
for type in rad instruct combine; do
  for train_dataset in dataset_RA_n479_n400 dataset_RA_n479_n400_n100; do
    for test_dataset in dataset_R_n139 dataset_RA_n479_n79 dataset_RAD_n102; do
      uv run python test.py \
        --seeds 0 1 2 3 4 \
        --train-dataset datasets/$train_dataset.pkl \
        --test-dataset datasets/$test_dataset.pkl \
        --model-dir storage/$type/$train_dataset \
        --n $N \
        --type $type \
        --csv >> $CSV
    done
  done
  uv run python test.py \
    --seeds 0 1 2 3 4 \
    --train-dataset datasets/dataset_RA_n479_n400.pkl \
    --test-dataset datasets/dataset_RA_n479_n400.pkl \
    --model-dir storage/$type/dataset_RA_n479_n400 \
    --n $N \
    --type $type \
    --csv >> $CSV
  uv run python test.py \
    --seeds 0 1 2 3 4 \
    --train-dataset datasets/dataset_RA_n479_n400_n100.pkl \
    --test-dataset datasets/dataset_RA_n479_n400_n100.pkl \
    --model-dir storage/$type/dataset_RA_n479_n400_n100 \
    --n $N \
    --type $type \
    --csv >> $CSV
done
