#!/usr/bin/env bash
python nnAvicaching_find_rewards.py --test "./stats/find_rewards/proportionateRewards_7-1_1747.txt" --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"

python nnAvicaching_find_rewards.py --test "./stats/find_rewards/randomRewards_sum_1000_6-29_1109.txt" --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"

python nnAvicaching_find_rewards.py --test "./stats/find_rewards/randomRewards_sum_1000_6-29_1203.txt" --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"

python nnAvicaching_find_rewards.py --test "./stats/find_rewards/randomRewards_sum_1000_7-3_1048.txt" --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"
