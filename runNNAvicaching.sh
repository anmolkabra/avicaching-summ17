#!/usr/bin/env bash
for lr in 0.1 0.01 0.001 0.0001 0.00001
do
    for mom in 0.1 0.3 0.5 0.7 0.9
    do
        python nnAvicaching.py --no-cuda --lr $lr --epochs 200 --momentum $mom --rand-xyr --save-plot
    done
done
