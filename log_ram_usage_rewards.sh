#!/usr/bin/env bash

# =============================================================================
# log_ram_usage_rewards.sh
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Log CPU, RAM, GPU usage when the Pricing Problem is run. Tests may change 
#   and script is dynamic.
# -----------------------------------------------------------------------------
# Required Dependencies/Software (others come pre-installed in Linux env):
#   - taskset
#   - nvidia-smi (GPU's poller)
# -----------------------------------------------------------------------------
# Required Local Files/Data/Modules:
#   - ./test_lp_time.py
# =============================================================================

# only lp
# echo "pid, cpu, mem, process" > ./stats/onlylp.txt
# wait $(jobs -p)
# python test_lp_time.py --epochs 200 &
# while [ `pgrep python` ]
# do
#     sleep .1
#     ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> ./stats/onlylp.txt && echo "----" >> ./stats/onlylp.txt
#     echo "onlylp"
# done
# exit

# gpu set
# echo "pid, cpu, mem, process" > ./stats/cpu_gpuset.txt
# wait $(jobs -p)
# python test_lp_time.py --epochs 200 &
# while [ `pgrep python` ]
# do
#     sleep .1
#     ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> ./stats/cpu_gpuset.txt && echo "----" >> ./stats/cpu_gpuset.txt
#     nvidia-smi >> ./stats/gpu_gpuset.txt
#     echo "gpuset"
# done

# cpu set
for th in 1 3 5 7
do
    echo "pid, cpu, mem, process" > "./stats/${th}cpu_cpuset.txt"
    wait $(jobs -p)
    taskset -c 1-$th python test_lp_time.py --epochs 200 --threads $th --no-cuda &
    while [ `pgrep python` ]
    do
        sleep .1
        ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> "./stats/${th}cpu_cpuset.txt" && echo "----" >> "./stats/${th}cpu_cpuset.txt"
        echo "${th}cpuset"
    done
done
