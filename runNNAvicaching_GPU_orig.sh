#!/usr/bin/env bash
for lambda in 1.0 5.0 10.0
do
        python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot
done