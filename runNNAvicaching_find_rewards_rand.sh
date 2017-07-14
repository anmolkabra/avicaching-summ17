# #!/usr/bin/env bash

# specs for tests:
# batch-size locations: 11, 35, 55, 85, 116
# T: 173
# lr: 0.001
# epochs: 1000

for J in 11 35 55 85 116
do
    for s in 1 2 3
    do
        python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/rand/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=119.2222 sec.txt" --locations $J --seed $s
        python nnAvicaching_find_rewards.py --rand --epochs 1000 --hide-loss-plot --weights-file "./stats/find_weights/weights/rand/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=119.2222 sec.txt" --locations $J --seed $s --no-cuda
    done
done
