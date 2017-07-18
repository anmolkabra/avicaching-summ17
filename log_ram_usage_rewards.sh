#!/usr/bin/env bash

# gpu set
python nnAvicaching_find_rewards.py --rand --epochs 25 --hide-loss-plot --weights-file "./stats/find_weights/weights/rand/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=119.2222 sec.txt" &
while [ `pgrep python` ]
do
    sleep .2
    ps aux | awk '{print $2, $4, $11}' | sort -k2rn | head -n 10 >> ./stats/cpu_gpuset.txt && echo "----" >> ./stats/cpu_gpuset.txt
    nvidia-smi >> ./stats/gpu_gpuset.txt
    echo "gpuset"
done

# cpu set
python nnAvicaching_find_rewards.py --rand --epochs 25 --hide-loss-plot --weights-file "./stats/find_weights/weights/rand/gpu, randXYR_seed=1, epochs=1000, train= 80%, lr=1.000e-03, time=119.2222 sec.txt" --no-cuda &
while [ `pgrep python` ]
do
    sleep .2
    ps aux | awk '{print $2, $4, $11}' | sort -k2rn | head -n 10 >> ./stats/cpu_cpuset.txt && echo "----" >> ./stats/cpu_cpuset.txt
    echo "cpuset"
done
