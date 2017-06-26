#!/usr/bin/env bash
nvidia-smi > ./stats/before_find_weights_gpu_status.txt
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 1
nvidia-smi > ./stats/running_find_weights_gpu_status.txt
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 2
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 3
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 1 --no-cuda
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 2 --no-cuda
python nnAvicaching_find_weights.py --epochs 10000 --hide-loss-plot --hide-map-plot --seed 3 --no-cuda
