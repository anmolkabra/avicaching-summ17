#!/usr/bin/env bash
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 1
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 2
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 3
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 1 --no-cuda
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 2 --no-cuda
python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 3 --no-cuda
