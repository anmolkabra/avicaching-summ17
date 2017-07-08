#!/usr/bin/env bash
for l in 0.01 0.001 0.0001 0.00001
do
    python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 1 --lr $l
    python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 2 --lr $l
    python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 3 --lr $l
    python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 4 --lr $l
    python nnAvicaching_find_weights.py --hide-map-plot --hide-loss-plot --epochs 10000 --seed 5 --lr $l
done

for l in 0.01 0.001 0.0001 0.00001
do
    python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed 1 --lr $l
    python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed 2 --lr $l
    python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed 3 --lr $l
    python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed 4 --lr $l
    python nnAvicaching_find_weights_hiddenlayer.py --no-plots --epochs 10000 --seed 5 --lr $l
done