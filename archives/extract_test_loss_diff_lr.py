#!/usr/bin/env python
from __future__ import print_function
import numpy as np, glob

def get_test_losses(fname):
    test_losses = []
    with open(fname, "r") as f:
        next(f)
        for idx, line in enumerate(f):
            if idx % 10 == 0:
                pos_test_loss = line.find("testloss = ") + 11
                test_losses.append(float(line[pos_test_loss:line.find(",", pos_test_loss)]))
    test_losses = np.array(test_losses)
    return test_losses

files = glob.glob("./stats/find_weights/logs/orig/*seed=5*")
files.sort()
all_losses = []
for idx, file in enumerate(files):
    losses = get_test_losses(file)
    if idx == 0:
        all_losses = losses
    else:
        all_losses = np.vstack( [all_losses, losses] )
loss_with_epoch = np.hstack( [np.arange(1, 10001, 10).reshape(1000, 1), all_losses.T] )
np.savetxt("test_losses_seed_5_diff_lr.csv", loss_with_epoch, fmt="%.4f", delimiter=",")