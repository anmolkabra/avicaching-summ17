#!/usr/bin/env python

# =============================================================================
# nnAvicaching_find_rewards.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Refer to the Report (link) for detailed explanation. In a gist, this script 
#   redistributes rewards in a budget over locations such that maximum 
#   homogenity is achieved. This is the step after finding weights in the 
#   Avicaching game. This model uses learned weights from the **3-layered 
#   network only**.
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - PyTorch
#   - NumPy
#   - SciPy
# -----------------------------------------------------------------------------
# Required Local Files/Data/Modules:
#   - ./data/*
#   - ./avicaching_data.py
#   - ./lp.py
#   - learned weights stored in a file in the format specified in avicaching_data
# =============================================================================

from __future__ import print_function
import numpy as np
import argparse
import time
import os
import sys
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as e:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import avicaching_data as ad
import lp
# import torch modules
import torch, torch.nn as nn
import torch.nn.functional as torchfun
import torch.optim as optim
from torch.autograd import Variable
matplotlib.rcParams.update({'font.size': 14})   # font-size for plots

# =============================================================================
# training options
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding rewards")
# training parameters
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
    help="inputs learning rate of the network (default=0.001)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
# data options
parser.add_argument("--locations", type=int, default=116, metavar="J",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=173, metavar="T",
    help="inputs total time of data collection; number of weeks (default=173)")
parser.add_argument("--rewards", type=float, default=1000.0, metavar="R",
    help="inputs the total budget of rewards to be distributed (default=1000.0)")
parser.add_argument("--rand", action="store_true", default=False,
    help="uses random data")
parser.add_argument("--weights-file", type=str, 
    default="./stats/find_weights/weights/orig/gpu, origXYR_seed=2, epochs=10000, train= 80%, lr=1.000e-03, time=1157.1230 sec.txt", 
    metavar="f", help="inputs the location of the file to use weights from")
parser.add_argument("--test", type=str, default="", 
    metavar="t", help="inputs the location of the file to test rewards from")
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='seed (default=1)')
# plot/log options
parser.add_argument("--hide-loss-plot", action="store_true", default=False,
    help="hides the loss plot, which is only saved")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
# deprecated options -- not deleting if one chooses to use them
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="[see script] inputs parameter eta in the model (default=10.0)")
parser.add_argument("--momentum", type=float, default=1.0, metavar="M",
    help="[see script] inputs SGD momentum (default=1.0)")   # if using SGD

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set the seeds
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# parameters and constants
# =============================================================================
# global values and datasets
torchten = torch.FloatTensor     # change here to use diff containers
J, T, totalR, numFeatures = args.locations, args.time, args.rewards, 0
weights_file_name = args.weights_file
X, w1_for_r, w2, F_DIST_w1 = [], [], [], []
lp_A, lp_c = [], []
loss = 0

# random datasets locations assigned to variables
locs_in_file = 232  # change this to use a diff random file
randXYR_file = "./data/random/randXYR" + str(locs_in_file) + ".txt"
randF_file = "./data/random/randF" + str(locs_in_file) + ".csv"
randDIST_file = "./data/random/randDIST" + str(locs_in_file) + ".txt"

# =============================================================================
# data input
# =============================================================================
def read_set_data():
    """
    Reads Datasets X, Y, f, D, weights from the files using avicaching_data 
    module's functions. f and D are then combined into F_DIST as preprocessed 
    tensor, which is then multiplied with w1 as preprocessing. All datasets are 
    normalized, expanded, averaged as required, leaving as torch tensors at the 
    end of the function.
    """
    global X, numFeatures, F_DIST_w1, w1_for_r, w2
    # shapes of datasets -- [] means expanded form:
    # - X, Y: J
    # - net.R: J [x J x 1]
    # - F_DIST: J x J x numF
    # - F_DIST_w1: J x J x numF
    # - w1: J x J x numF
    # - w2: J x numF [x 1]
    # - w1_for_r: J x 1 x numF

    # read f and DIST datasets from file, operate on them
    if args.rand:
        F = ad.read_F_file(randF_file, J)
        DIST = ad.read_dist_file(randDIST_file, J)
    else:
        F = ad.read_F_file(
            "./data/loc_feature_with_avicaching_combined.csv", J)
        DIST = ad.read_dist_file(
            "./data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt", 
            J)
    F = ad.normalize(F, along_dim=0, using_max=True)    # normalize using max
    DIST = ad.normalize(DIST, using_max=True)  # normalize using max
    
    # combine f and D for the NN
    numFeatures = len(F[0]) + 1     # compensating for the distance element
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 1                # for reward later

    # read weights and X
    w1, w2 = ad.read_weights_file(weights_file_name, locs_in_file, J, numFeatures)
    w2 = np.expand_dims(w2, axis=2)
    if args.rand:
        X, _, _ = ad.read_XYR_file(randXYR_file, J, T)
    else:
        X, _, _ = ad.read_XYR_file(
            "./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", 
            J, T)
    
    # split w1; multiply the F_DIST portion of w1 with F_DIST
    w1_for_fdist, w1_for_r = ad.split_along_dim(w1, numFeatures - 1, dim=1)
    F_DIST_w1 = Variable(torch.bmm(
        F_DIST, torchten(w1_for_fdist)), requires_grad=False)

    # condense X along T into a single vector and normalize
    X = ad.normalize(X.sum(axis=0), using_max=False)
    
    w1_for_r = Variable(torchten(w1_for_r), requires_grad=False)
    X = Variable(torchten(X), requires_grad=False)
    w2 = Variable(torchten(w2), requires_grad=False)

# =============================================================================
# PriProb class
# =============================================================================
class PriProb(nn.Module):
    """
    An instance of this class emulates the sub-model used in Pricing Problem to 
    calculate the loss function and update rewards. Constraining not done here.
    """

    def __init__(self):
        """Initializes PriProb, creates the rewards dataset for the model."""
        super(PriProb, self).__init__()
        # initialize R: distribute totalR reward points in J locations randomly
        # self.r preserved for debugging, no real use in the script
        self.r = np.random.multinomial(totalR, [1 / float(J)] * J, size=1)
        normalizedR = ad.normalize(self.r, using_max=False)
        self.R = nn.Parameter(torchten(normalizedR))

    def forward(self, wt1, wt2):
        """
        Goes forward in the network -- multiply the weights, apply relu, 
        multiply weights again and apply softmax

        Returns:
            torch.Tensor -- result after going forward in the network.
        """
        repeatedR = self.R.repeat(J, 1).unsqueeze(dim=2)    # shape is J x J x 1
        # multiply wt1 with r and add the resulting tensor with the already 
        # calculated F_DIST_w1. Drastically improves performance. 
        
        # If you've trouble understanding why multiply and add, draw these 
        # tensors on a paper and work out how the additions and multiplications 
        # affect elements, i.e., which operations affect which sections of the 
        # tensors 
        res = torch.bmm(repeatedR, wt1) + F_DIST_w1     # res is J x J x numF after
        # forward propagation done, multiply remaining tensors (no tensors are 
        # mutable after this point except res)
        res = torchfun.relu(res)
        res = torch.bmm(res, wt2).view(-1, J)    # res is J x J
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        return torchfun.softmax(res)

def go_forward(net):
    """
    Feed forward the dataset in the model's network and calculate Y and loss.

    Args:
        net -- (PriProb instance)

    Returns:
        float -- time taken to go complete all operations in the network
    """
    global w1_for_r, w2, loss
    start_forward_time = time.time()

    # feed in data
    P = net(w1_for_r, w2).t()
    # calculate loss
    Y = torch.mv(P, X)
    loss = torch.norm(Y - torch.mean(Y).expand_as(Y)).pow(2) / J
    
    return time.time() - start_forward_time

def train(net, optimizer):
    """
    Trains the Neural Network using PriProb on the training set.

    Args:
        net -- (PriProb instance)
        optimizer -- (torch.optim instance) Gradient-Descent function
        
    Returns:
        3-tuple -- (Execution Time, End loss value, 
            Model's prediction after feed forward [Px])
    """
    global lp_A, lp_c, loss

    # BACKPROPAGATE
    start_backprop_time = time.time()
    
    optimizer.zero_grad()
    loss.backward()         # calculate grad
    optimizer.step()        # update rewards
    
    r_on_cpu = net.R.data.squeeze().cpu().numpy()   # transfer data for lp
    backprop_time = time.time() - start_backprop_time

    start_lp_time = time.time()
    # CONSTRAIN -- LP
    # 1.0 is the sum constraint of rewards
    # the first J outputs are the new rewards
    net.R.data = torchten(lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0).x[:J]).unsqueeze(dim=0)
    lp_time = time.time() - start_lp_time

    trans_time = time.time()
    if args.cuda:
        # transfer data
        net.R.data = net.R.data.cuda()
    trans_time = time.time() - trans_time
    
    # FORWARD
    forward_time = go_forward(net)

    return (backprop_time + lp_time + forward_time + trans_time, lp_time)

# =============================================================================
# logs and plots
# =============================================================================
def save_log(file_name, results, title, rewards=None):
    """
    Saves the log to a file.

    Args:
        file_name -- (str) name of the file
        results -- (tuple) time taken for model run and end loss value
        title -- (str) title of the plot
        rewards -- (NumPy ndarray or None) rewards (default=None)
    """
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        f.write("J: %3d\t\tT: %3d\n-------------\n" % (J, T))
        if rewards is not None:
            np.savetxt(f, rewards, fmt="%.15f", delimiter=" ")
        f.write("time = %.4f\t\tloss = %.15f\n" % (results[0], results[1]))

def save_plot(file_name, x, y, xlabel, ylabel, title):
    """
    Saves and (optionally) shows the loss plot of train and test periods.

    Args:
        file_name -- (str) name of the file
        x -- (NumPy ndarray) data on the x-axis
        y -- (NumPy ndarray) data on the y-axis
        xlabel -- (str) label for the x-axis
        ylabel -- (str) what else can it mean?
        title -- (str) title of the plot
    """
    # plot details
    loss_fig = plt.figure(1)
    train_label, = plt.plot(x, y, "r-", label="Train Loss")
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.title(title)

    # save and show
    loss_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_loss_plot:
        plt.show()
    plt.close()

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    read_set_data()
    net = PriProb()
    transfer_time = time.time()
    if args.cuda:
        net.cuda()
        w1_for_r, w2, F_DIST_w1, X = w1_for_r.cuda(), w2.cuda(), F_DIST_w1.cuda(), X.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    transfer_time = time.time() - transfer_time
    
    if args.test:
        # secondary function of the script -- calculate loss value for the 
        # supplied data
        rewards = np.loadtxt(args.test, delimiter=" ")[:J]
        rewards = torchten(ad.normalize(rewards, using_max=False))
        if args.cuda:
            rewards = rewards.cuda()
        net.R.data = rewards        # substitute manual rewards
        forward_time = go_forward(net)
        # save results
        fname = "testing \"" + args.test[args.test.rfind("/") + 1:] + '"' + str(time.time())
        save_log("./stats/find_rewards/test_rewards_results/" + fname + ".txt", 
            (forward_time, loss.data[0]), weights_file_name)
        sys.exit(0)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lp_A, lp_c = lp.build_A(J), lp.build_c(J)

    # refer to the report and Algorithm for Pricing Problem to understand the 
    # logic flow: forward -> loop [backpropagate -> update -> constrain -> forward]
    best_loss, total_lp_time = float("inf"), 0
    total_time = go_forward(net)    # start model and logging here
    total_time += transfer_time
    train_loss = [ loss.data[0] ]

    for e in xrange(1, args.epochs + 1):
        train_t = train(net, optimizer)
        curr_loss = loss.data[0]
        train_loss.append(curr_loss)

        if curr_loss < best_loss:
            # save the best result uptil now
            best_loss = curr_loss
            best_rew = net.R.data.clone()
        
        total_time += train_t[0]
        total_lp_time += train_t[1]
        if e % 20 == 0:
            print("epoch=%5d, loss=%.10f, budget=%.10f" % \
                (e, curr_loss, net.R.data.sum()))
    best_rew = best_rew.cpu().numpy() * totalR  # de-normalize rewards

    # log and plot the results: epoch vs loss

    # define file names
    if args.rand:
        file_pre = "randXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
    else:
        file_pre = "origXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
    log_name = "lr=%.3e, bestloss=%.6f, time=%.4f sec, lp_time=%.4f sec" % (
        args.lr, best_loss, total_time, total_lp_time)
    epoch_data = np.arange(0, args.epochs + 1)
    fname = file_pre_gpu + file_pre + log_name
    # save amd plot data
    save_plot("./stats/find_rewards/plots/" + fname + ".png", epoch_data, 
        train_loss, "epoch", "loss", log_name)
    save_log("./stats/find_rewards/logs/" + fname + ".txt", (total_time, best_loss),
        weights_file_name, best_rew)

    print("---> " + fname + " DONE")
