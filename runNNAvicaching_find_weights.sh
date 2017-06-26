#!/usr/bin/env bash
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 1
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 2
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 3
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 1 --no-cuda
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 2 --no-cuda
python nnAvicaching.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 3 --no-cuda