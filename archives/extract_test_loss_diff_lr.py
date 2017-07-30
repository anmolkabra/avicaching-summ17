#!/usr/bin/env python

# =============================================================================
# extract_test_loss_diff_lr.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Creates a csv file of combining test losses from logs of 
#   nnAvicaching_find_weights.py runs. The test losses are collected from all 
#   log files of seed 5 and diff learning rates. The csv is used to plot Loss vs 
#   Epoch plot for different learning rates in Section 4.1.1 of the report
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - NumPy
# -----------------------------------------------------------------------------
# Required Local Files/Data/Modules:
#   - ./stats/find_weights/logs/orig/*seed=5* (mutable)
# =============================================================================

from __future__ import print_function
import numpy as np
import glob

def get_test_losses(fname):
    """
    Returns the test losses extracted from file.

    Args:
        fname -- name of the file

    Returns:
        NumPy ndarray -- of every 10th test loss (otherwise 10000 test loss for 
            each learning rate becomes hard to handle for tikz in latex)
    """
    test_losses = []
    with open(fname, "r") as f:
        next(f) # first line doesn't contain anything
        for idx, line in enumerate(f):
            if idx % 10 == 0:
                # only collect every 10th log in the file 
                # get the test loss value
                pos_test_loss = line.find("testloss = ") + 11
                test_losses.append(float(line[pos_test_loss:line.find(",", pos_test_loss)]))
    test_losses = np.array(test_losses)
    return test_losses

files = glob.glob("./stats/find_weights/logs/orig/*seed=5*")
files.sort()    # sorts the list into decreasing order of learning rates
all_losses = []
for idx, file in enumerate(files):
    losses = get_test_losses(file)
    if idx == 0:
        all_losses = losses
    else:
        all_losses = np.vstack( [all_losses, losses] )  # append to form a matrix

# prepend the epoch to the all_losses matrix
# every 10th log is collected, which represents every 10th epoch
loss_with_epoch = np.hstack( [np.arange(1, 10001, 10).reshape(1000, 1), all_losses.T] )
np.savetxt("test_losses_seed_5_diff_lr.csv", loss_with_epoch, fmt="%.4f", delimiter=",")