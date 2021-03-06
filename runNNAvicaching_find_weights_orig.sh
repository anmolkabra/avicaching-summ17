#!/usr/bin/env bash

# =============================================================================
# runNNAvicaching_find_weights_orig.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Script to run multiple versions (diff seeds/features for model) of the 
#   Identification Problem's model in nnAvicaching_find_weights.py and 
#   nnAvicaching_find_weights_hiddenlayer.py. Optimization on original datasets.
# =============================================================================

# specs for tests:
# seeds: 1, 2, 3, 4, 5
# lr: 10^{-2, -3, -4, -5}
# J: 116
# T: 173
# epochs: 10000

for l in 0.01 0.001 0.0001 0.00001
do
    for s in 1 2 3 4 5
    do
        # 3-layered model with GPU "set"
        python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed $s --lr $l
        # 4-layered model with GPU "set"
        python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed $s --lr $l
    done
done
