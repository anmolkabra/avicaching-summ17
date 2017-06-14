#!/usr/bin/env bash
for e in 200 400 600
do
    for lr in 0.1 0.01
    do
        python nnAvicaching.py --lr $lr --epochs $e --train-percent 1.0 --rand-xyr --save-plot
    done
done
