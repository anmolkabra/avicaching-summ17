#!/usr/bin/env bash
python nnAvicaching_find_rewards.py --seed 1 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-02, time=981.7946 sec.txt"
python nnAvicaching_find_rewards.py --seed 2 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-02, time=983.9379 sec.txt"
python nnAvicaching_find_rewards.py --seed 3 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=3, epochs=10000, train= 80%, lr=1.000e-02, time=975.6399 sec.txt"
python nnAvicaching_find_rewards.py --seed 1 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-02, time=981.7946 sec.txt" --no-cuda
python nnAvicaching_find_rewards.py --seed 2 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-02, time=983.9379 sec.txt" --no-cuda
python nnAvicaching_find_rewards.py --seed 3 --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=3, epochs=10000, train= 80%, lr=1.000e-02, time=975.6399 sec.txt" --no-cuda