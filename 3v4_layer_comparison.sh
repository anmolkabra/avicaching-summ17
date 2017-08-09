#!/usr/bin/env bash

# =============================================================================
# 3v4_layers_comparison.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Script to test if 4-layer network performs better than 3-layer when the 
#   are bigger. Using random dataset as original dataset doesn't contain more 
#   than 116 locations.
# =============================================================================

# specs for tests:
# seeds: 1
# lr: 10^-4
# J: 116, 232
# T: 173
# epochs: 10000

for J in 116 232
do
    python nnAvicaching_find_weights.py --rand --locations $J --hide-map-plot --hide-loss-plot --epochs 10000 --seed 1 --lr 1e-4
    python nnAvicaching_find_weights_hiddenlayer.py --rand --locations $J --hide-map-plot --hide-loss-plot --epochs 10000 --seed 1 --lr 1e-4
done
