#!/usr/bin/env bash
for s in 1 2 3 4 5
do
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-02, time=1260.1639 sec.txt"
done
for s in 1 2 3 4 5
do
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-02, time=1262.2267 sec.txt"
done
for s in 1 2 3 4 5
do
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=3, epochs=10000, train= 80%, lr=1.000e-02, time=1261.1577 sec.txt"
done
for s in 1 2 3 4 5
do
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=4, epochs=10000, train= 80%, lr=1.000e-02, time=1259.9575 sec.txt"
done
for s in 1 2 3 4 5
do
    python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/gpu, origXYR_seed=5, epochs=10000, train= 80%, lr=1.000e-02, time=1259.5761 sec.txt"
done
