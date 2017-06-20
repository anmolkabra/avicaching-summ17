#!/usr/bin/env bash
for lam in 1.0 5.0
do
        python nnAvicaching.py --epochs 10000 --lambda-L1 $lam --hide-loss-plot --hide-map-plot
done