#!/usr/bin/env bash
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 1
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 2
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 3
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 1 --no-cuda
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 2 --no-cuda
python nnAvicaching_find_weights.py --rand-xyr --epochs 1000 --hide-loss-plot --hide-map-plot --seed 3 --no-cuda
