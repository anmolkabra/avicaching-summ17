#!/usr/bin/env bash

# specs for tests:
# locations: 116
# batch-size T: 17, 51, 85, 129, 173
# lr: 0.001
# epochs: 1000

for T in 17 51 85 129 173
do
    for s in 1 2 3
    do
        taskset -c 3 python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --time $T --seed $s
        taskset -c 4 python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --time $T --seed $s --no-cuda
    done
done
