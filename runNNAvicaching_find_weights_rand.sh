#!/usr/bin/env bash

# =============================================================================
# runNNAvicaching_find_weights_rand.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Script to run multiple versions (diff seeds/features for model) of the 
#   Identification Problem's model in nnAvicaching_find_weights.py. 
#   GPU Speedup on random datasets.
# =============================================================================

# specs for tests:
# T: 173
# batch-size J: 11, 37, 63, 90, 116, 145, 174, 203, 232
# lr: 0.001
# epochs: 1000

for J in 11 37 63 90 116 145 174 203 232
do
    for s in 1 2 3
    do
        # 3-layered model only. time not recorded for the 4-layered model
        python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --locations $J --seed $s
        python nnAvicaching_find_weights.py --rand --no-plots --epochs 1000 --locations $J --seed $s --no-cuda
    done
done
