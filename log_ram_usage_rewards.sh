#!/usr/bin/env bash

# only lp
echo "pid, cpu, mem, process" > ./stats/onlylp.txt
wait $(jobs -p)
python test_lp_time.py --epochs 200 &
while [ `pgrep python` ]
do
    sleep .2
    ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> ./stats/onlylp.txt && echo "----" >> ./stats/onlylp.txt
    echo "onlylp"
done
exit

# gpu set
echo "pid, cpu, mem, process" > ./stats/cpu_gpuset.txt
wait $(jobs -p)
python test_lp_time.py --epochs 200 &
while [ `pgrep python` ]
do
    sleep .2
    ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> ./stats/cpu_gpuset.txt && echo "----" >> ./stats/cpu_gpuset.txt
    nvidia-smi >> ./stats/gpu_gpuset.txt
    echo "gpuset"
done

# cpu set
echo "pid, cpu, mem, process" > ./stats/cpu_cpuset.txt
wait $(jobs -p)
python test_lp_time.py --epochs 200 --no-cuda &
while [ `pgrep python` ]
do
    sleep .2
    ps aux | awk '{print $2, $3, $4, $11}' | sort -k2rn | head -n 3 >> ./stats/cpu_cpuset.txt && echo "----" >> ./stats/cpu_cpuset.txt
    echo "cpuset"
done
