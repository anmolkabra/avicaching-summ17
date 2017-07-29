#!/usr/bin/env bash

# =============================================================================
# runNNAvicaching_find_rewards_orig.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Script to run multiple versions (diff seeds/features for model) of the 
#   Pricing Problem's model in nnAvicaching_find_rewards.py. Optimization on 
#   original datasets.
# =============================================================================

# specs for tests:
# seeds: 1, 2, 3, 4, 5
# J: 116
# T: 173
# lr: 0.001
# epochs: 1000

for s in 1 2 3 4 5
do
    # determine losses with differently seeded rewards on diff sets of weights
    # only GPU "set"
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-03, time=1164.3672 sec.txt"
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=3, epochs=10000, train= 80%, lr=1.000e-03, time=1190.5355 sec.txt"
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=4, epochs=10000, train= 80%, lr=1.000e-03, time=1170.7300 sec.txt"
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=5, epochs=10000, train= 80%, lr=1.000e-03, time=1438.7272 sec.txt"
done

# find the best rewards knowing that a weights file gives best average results
# weights file changed according to the performance of the set of weights. In 
# our tests, set-2 of weights (seeded 2) was consistently performing better.

# for l in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001
# do
#     for s in 1 2 3 4 5
#     do
#         python nnAvicaching_find_rewards.py --lr $l --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"
#     done
# done
