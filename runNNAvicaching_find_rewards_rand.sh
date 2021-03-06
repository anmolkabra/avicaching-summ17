#!/usr/bin/env bash

# =============================================================================
# runNNAvicaching_find_rewards_rand.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Script to run multiple versions (diff batch-sizes/seeds for model) of the 
#   Pricing Problem's model in nnAvicaching_find_rewards.py. GPU Speedup on 
#   random datasets
# =============================================================================

# specs for tests:
# batch-size locations: 11, 37, 63, 90, 116, 145, 174, 203, 232
# T: 173
# lr: 0.001
# epochs: 1000

for J in 11 37 63 90 116 145 174 203 232
do
    for s in 1 2 3
    do
        python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=271.2067 sec.txt" --locations $J --seed $s
        python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=271.2067 sec.txt" --locations $J --seed $s --no-cuda
    done
done

# restrict cpu access
# change to-be-saved file name
sed -i 's%find_rewards/plots/%find_rewards/plots/1_%g' nnAvicaching_find_rewards.py
sed -i 's%find_rewards/logs/%find_rewards/logs/1_%g' nnAvicaching_find_rewards.py
for J in 11 37 63 90 116 145 174 203 232
do
    for s in 1 2 3
    do
        taskset -c 1 python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=271.2067 sec.txt" --locations $J --seed $s
        taskset -c 2 python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=271.2067 sec.txt" --locations $J --seed $s --no-cuda
    done
done
# revert to normal
sed -i 's%find_rewards/plots/1_%find_rewards/plots/%g' nnAvicaching_find_rewards.py
sed -i 's%find_rewards/logs/1_%find_rewards/logs/%g' nnAvicaching_find_rewards.py
