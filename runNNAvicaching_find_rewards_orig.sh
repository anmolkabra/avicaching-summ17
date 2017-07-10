#!/usr/bin/env bash
# for s in 1 2 3 4 5
# do
#     python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=1, epochs=10000, train= 80%, lr=1.000e-03, time=1164.3672 sec.txt"
# done
# for s in 1 2 3 4 5
# do
#     python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"
# done
# for s in 1 2 3 4 5
# do
#     python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=3, epochs=10000, train= 80%, lr=1.000e-03, time=1190.5355 sec.txt"
# done
# for s in 1 2 3 4 5
# do
#     python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=4, epochs=10000, train= 80%, lr=1.000e-03, time=1170.7300 sec.txt"
# done
# for s in 1 2 3 4 5
# do
#     python nnAvicaching_find_rewards.py --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=5, epochs=10000, train= 80%, lr=1.000e-03, time=1438.7272 sec.txt"
# done

# find the best rewards knowing that a weights file gives best av results
for l in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001
do
    for s in 1 2 3 4 5
    do
        python nnAvicaching_find_rewards.py --lr $l --seed $s --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt"
    done
done
