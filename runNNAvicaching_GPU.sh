#!/usr/bin/env bash
for lr in 0.1 0.01 0.001
do
    for mom in 0.5 1.0 1.5 2.0 2.5 3.0
    do
        python nnAvicaching.py --lr $lr --epochs 200 --momentum $mom --rand-xyr --save-plot
    done
done
