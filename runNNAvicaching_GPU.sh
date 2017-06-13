#!/usr/bin/env bash
for (( e = 20; e <= 200; e += 20 ))
do
    for lr in 0.1 0.01 0.001 0.0001 0.00001
    do
        python nnAvicaching.py --lr $lr --epochs $e --rand-xyr --momentum 0.5 --eta 5 --lambda-L1 5 --save-plot
    done
done
