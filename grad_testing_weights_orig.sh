#!/usr/bin/env bash

python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 100 --seed 1 --locations 11 --lr 0.001
python nnAvicaching_find_weights_hiddenlayer.py --hide-map-plot --hide-loss-plot --epochs 100 --seed 1 --locations 11 --lr 0.001