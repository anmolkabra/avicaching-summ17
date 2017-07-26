#!/usr/bin/env bash

# testing GPU "set" with half and float tensors
# specs for tests:
# locations: 116
# batch-size T: 17, 51, 85, 129, 173
# lr: 0.001
# epochs: 1000

for T in 17 51 85 129 173
do
    for s in 1 2 3
    do
        # sed -i 's/HalfTensor/FloatTensor/g' nnAvicaching_find_weights.py
        # python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --time $T --seed $s
        sed -i 's/FloatTensor/HalfTensor/g' nnAvicaching_find_weights.py
        python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --time $T --seed $s
    done
done

python nnAvicaching_find_weights.py --no-plots --epochs 10000 --seed 1 --lr 0.001
