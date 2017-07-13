#!/usr/bin/env bash

# specs for tests:
# locations: 220
# batch-size T: 17, 51, 85, 129, 173, 255, 340
# lr: 0.001
# epochs: 1000

for T in 17 51 85 129 173 255 340
do
    for s in 1 2 3
    do
        python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --locations 220 --time $T --seed $s
        python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --locations 220 --time $T --seed $s --no-cuda
    done
done