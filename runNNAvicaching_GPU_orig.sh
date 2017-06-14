#!/usr/bin/env bash
for e in 200 400 600
do
    for lr in 0.1 0.01
    do
        python nnAvicaching.py --lr $lr --epochs $e --save-plot
    done
done
