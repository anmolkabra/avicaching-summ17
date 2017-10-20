#!/usr/bin/env python

# =============================================================================
# nnAvicaching_find_weights_hiddenlayer.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Refer to the Report (link) for detailed explanation. In a gist, this script
#   learns the weights that highlight the change of eBird agents' behavior
#   after certain rewards are applied. The model uses a **4-layered** neural
#   network.
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - PyTorch
#   - NumPy
# -----------------------------------------------------------------------------
# Required Local Files/Data/Modules:
#   - ./data/*
#   - ./avicaching_data.py
# =============================================================================

from __future__ import print_function
import argparse
import time
import math
import os
import sys
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as e:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import avicaching_data as ad
# import torch modules
import torch, torch.nn as nn
import torch.nn.functional as torchfun
import torch.optim as optim
from torch.autograd import Variable
matplotlib.rcParams.update({'font.size': 14})   # font-size for plots

# =============================================================================
# training specs
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding weights")
# training parameters
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
    help="inputs learning rate of the network (default=0.001)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
# data options
parser.add_argument("--train-percent", type=float, default=0.8, metavar="T",
    help="breaks the data into T percent training and rest testing (default=0.8)")
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='random seed (default=1)')
parser.add_argument("--locations", type=int, default=116, metavar="J",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=173, metavar="T",
    help="inputs total time of data collection; number of weeks (default=173)")
parser.add_argument("--rand", action="store_true", default=False,
    help="uses random xyr data")
# plot/log options
parser.add_argument("--no-plots", action="store_true", default=False,
    help="skips generating plot maps")
parser.add_argument("--hide-loss-plot", action="store_true", default=False,
    help="hides the loss plot, which is only saved")
parser.add_argument("--hide-map-plot", action="store_true", default=False,
    help="hides the map plot, which is only saved")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
# deprecated options -- not deleting if one chooses to use them
parser.add_argument("--expand-R", action="store_true", default=False,
    help="[see script] expands the reward vectors into matrices with distributed rewards")
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="[see script] inputs parameter eta in the model (default=10.0)")
parser.add_argument("--lambda-L1", type=float, default=10.0, metavar="LAM",
    help="[see script] inputs the L1 regularizing coefficient")
parser.add_argument("--momentum", type=float, default=1.0, metavar="M",
    help="[see script] inputs SGD momentum (default=1.0)")   # if using SGD

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.should_test = (args.train_percent != 1.0)

# set the seeds
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# constants and parameters
# =============================================================================
# global values and datasets
torchten = torch.FloatTensor    # change here to use diff containers
J, T, numFeatures = args.locations, args.time, 0
trainX, trainY, trainR, testX, testY, testR, F_DIST = [], [], [], [], [], [], []
u_train, u_test = np.array([]), np.array([])

num_train = int(math.floor(args.train_percent * T))
num_test = T - num_train

# random datasets locations assigned to variables
locs_in_file = 232  # change this to use a diff random file
randXYR_file = "./data/random/randXYR" + str(locs_in_file) + ".txt"
randXYR_weights_file = "./data/random/randXYR" + str(locs_in_file) + "_weights.txt"
randF_file = "./data/random/randF" + str(locs_in_file) + ".csv"
randDIST_file = "./data/random/randDIST" + str(locs_in_file) + ".txt"

# =============================================================================
# data input functions
# =============================================================================
def read_set_data():
    """
    Reads Datasets X, Y, R, f, D from the files using avicaching_data
    module's functions. f and D are then combined into F_DIST as preprocessed
    tensor. All datasets are normalized, expanded, averaged as required,
    leaving as torch tensors at the end of the function.
    """
    global trainX, trainY, trainR, testX, testY, testR, F_DIST, numFeatures
    global u_train, u_test
    # shapes of datasets -- [] means expanded form:
    # - X, Y: T x J
    # - R: T x J [x 15]
    # - net.w1: J x numF x numF
    # - net.w2: J x numF x numF
    # - net.w3: J x numF x numF
    # - net.w4: J x numF x 1
    # - F_DIST: J x J x numF

    # read f and DIST datasets from file, operate on them
    if args.rand:
        F = ad.read_F_file(randF_file, J)
        DIST = ad.read_dist_file(randDIST_file, J)
    else:
        F = ad.read_F_file(
            "../sensitive-avicaching/data/loc_feature_with_avicaching_combined.csv", J)
        DIST = ad.read_dist_file(
            "../sensitive-avicaching/data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt",
            J)
    F = ad.normalize(F, along_dim=0, using_max=True)    # normalize using max
    DIST = ad.normalize(DIST, using_max=True)   # normalize using max

    # process data for the NN
    numFeatures = len(F[0]) + 1     # compensating for the distance element
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 1                # for reward later

    # operate on XYR data
    X, Y, R = [], [], []
    if args.rand:
        if not os.path.isfile(randXYR_file):
            # file doesn't exists, make random data, write to file
            X, Y, R = make_rand_data()
            ad.save_rand_XYR(randXYR_file, X, Y, R, J, T)
        X, Y, R = ad.read_XYR_file(randXYR_file, J, T)
    else:
        X, Y, R = ad.read_XYR_file(
            "../sensitive-avicaching/data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt",
            J, T)

    u = np.sum(Y, axis=1)   # u weights for calculating losses

    # normalize X, Y using sum along rows
    X = ad.normalize(X, along_dim=1, using_max=False)
    Y = ad.normalize(Y, along_dim=1, using_max=False)
    if not args.expand_R:
        R = ad.normalize(R, along_dim=0, using_max=False)

    # split the XYR data
    if args.should_test:
        # training and testing, shuffle and split the data
        shuffle_order = np.random.permutation(T)
        trainX, testX = ad.split_along_dim(X[shuffle_order], num_train, dim=0)
        trainY, testY = ad.split_along_dim(Y[shuffle_order], num_train, dim=0)
        trainR, testR = ad.split_along_dim(R[shuffle_order], num_train, dim=0)
        u_train, u_test = ad.split_along_dim(u[shuffle_order], num_train, dim=0)
    else:
        # no testing, split the data -> test Matrices are empty
        trainX, testX = ad.split_along_dim(X, num_train, dim=0)
        trainY, testY = ad.split_along_dim(Y, num_train, dim=0)
        trainR, testR = ad.split_along_dim(R, num_train, dim=0)
        u_train, u_test = ad.split_along_dim(u, num_train, dim=0)

    # change the input data into pytorch tensors and variables
    trainR, testR = torchten(trainR), torchten(testR)
    u_train, u_test = torchten(u_train), torchten(u_test)
    trainX = Variable(torchten(trainX), requires_grad=False)
    trainY = Variable(torchten(trainY), requires_grad=False)
    testX = Variable(torchten(testX), requires_grad=False)
    testY = Variable(torchten(testY), requires_grad=False)

    if args.expand_R:
        # expand R (trainR and testR)
        trainR_ext = torchten(num_train, J, 15)
        testR_ext = torchten(num_test, J, 15)
        for t in xrange(num_train):
            trainR_ext[t] = expand_R(trainR[t], R_max=15)
        for t in xrange(num_test):
            testR_ext[t] = expand_R(testR[t], R_max=15)
        trainR, testR = trainR_ext, testR_ext
        numFeatures += 14   # 1 reward already added, adding the remaining 14

def make_rand_data(X_max=100.0, R_max=100.0):
    """
    This script uses the random datasets generated by
    nnAvicaching_find_weights.py (use random datasets only for measuring
    computation time -- results don't matter). So this function doesn't have
    much use.

    Creates random X and R and calculates Y based on random weights. Also
    stores the weights in files before returning.

    Args:
        X_max -- (float) Maximum value of element in X dataset (default=100.0)
        R_max -- (float) Maximum value of element in R dataset (default=100.0)

    Returns:
        3-tuple -- (X, Y, R) (values are not de-normalized)
    """
    global F_DIST
    # create random X and R and w
    origX = np.floor(np.random.rand(T, J) * X_max)
    origR = np.floor(np.random.rand(T, J) * R_max)
    X = ad.normalize(origX, along_dim=1, using_max=False)
    R = torchten(ad.normalize(origR, along_dim=0, using_max=False))
    w1 = Variable(torch.randn(J, numFeatures, numFeatures).type(torchten))
    w2 = Variable(torch.randn(J, numFeatures, numFeatures).type(torchten))
    w3 = Variable(torch.randn(J, numFeatures, numFeatures).type(torchten))
    w4 = Variable(torch.randn(J, numFeatures, 1).type(torchten))

    # convert to torch tensor and create placeholder for Y
    Y = np.empty([T, J])
    X = Variable(torchten(X), requires_grad=False)
    Y = Variable(torchten(Y), requires_grad=False)
    if args.cuda:
        # transfer to GPU
        X, Y, R, F_DIST = X.cuda(), Y.cuda(), R.cuda(), F_DIST.cuda()
        w1, w2, w3 = w1.cuda(), w2.cuda(), w3.cuda()

    # build Y
    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)

        # feed in data
        inp = torchfun.relu(torch.bmm(inp, w1)) # first weights
        inp = torchfun.relu(torch.bmm(inp, w2)) # second weights
        inp = torchfun.relu(torch.bmm(inp, w3)) # third weights
        inp = torch.bmm(inp, w4).view(-1, J)    # fourth weights
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        P = torchfun.softmax(inp).t()

        # calculate Y
        Y[t] = torch.mv(P, X[t])

    # for verification of random data, save weights ---------------------------
    w1_matrix = w1.data.cpu().numpy()
    w2_matrix = w2.data.cpu().numpy()
    w3_matrix = w3.data.cpu().numpy()
    w4_matrix = w4.data.view(-1, numFeatures).cpu().numpy()

    with open(randXYR_weights_file, "w") as f:
        # save w1
        f.write('# w1 shape: {0}\n'.format(w1.shape))
        for data_slice in w1_matrix:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")

        # save w2
        f.write('# w2 shape: {0}\n'.format(w2.shape))
        for data_slice in w2_matrix:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")
        
        # save w3
        f.write('# w3 shape: {0}\n'.format(w3.shape))
        for data_slice in w3_matrix:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")

        # save w4
        f.write('# w4 shape: {0}\n'.format(w4.shape))
        np.savetxt(f, w4_matrix, fmt="%.15f", delimiter=" ")
    # -------------------------------------------------------------------------

    return (X.data.cpu().numpy(), Y.data.cpu().numpy(), R.cpu().numpy())

def test_given_data(X, Y, R, w1, w2, w3, w4, J, T, u):
    """
    Tests a given set of datasets, printing the loss value after one
    forward propagation.

    Args:
        All arguments are self-explanatory
    """
    # loss_normalizer divides the calculated loss after feed forward
    # formula = || ((u * (Y-mean(Y)))^2 ||
    loss_normalizer = (torch.mv(torch.t(Y \
        - torch.mean(Y).expand_as(Y)).data, u)).pow(2).sum()
    loss = 0

    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)

        # feed in data
        inp = torchfun.relu(torch.bmm(inp, w1)) # first weights
        inp = torchfun.relu(torch.bmm(inp, w2)) # second weights
        inp = torchfun.relu(torch.bmm(inp, w3)) # third weights
        inp = torch.bmm(inp, w4).view(-1, J)    # fourth weights
        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        P = torchfun.softmax(inp).t()

        # calculate loss
        Pxt = torch.mv(P, X[t])
        loss += (u[t] * (Y[t] - Pxt)).pow(2).sum()
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer
    print("Loss = %f" % loss.data[0])

# =============================================================================
# IdProb4 class
# =============================================================================
class IdProb5(nn.Module):
    """
    An instance of this class emulates the model used for Identification
    Problem as a 4-layered network.
    """

    def __init__(self):
        """Initializes IdProb4, creates the sets of weights for the model."""
        super(IdProb5, self).__init__()
        self.w1 = nn.Parameter(torch.randn(J, numFeatures, numFeatures).type(
            torchten))
        self.w2 = nn.Parameter(torch.randn(J, numFeatures, numFeatures).type(
            torchten))
        self.w3 = nn.Parameter(torch.randn(J, numFeatures, numFeatures).type(
            torchten))
        self.w4 = nn.Parameter(torch.randn(J, numFeatures, 1).type(torchten))

    def forward(self, inp):
        """
        Goes forward in the network -- multiply the weights, apply relu,
        multiply weights again and apply softmax

        Returns:
            torch.Tensor -- result after going forward in the network.
        """
        inp = torchfun.relu(torch.bmm(inp, self.w1))    # first weights
        inp = torchfun.relu(torch.bmm(inp, self.w2))    # second weights
        inp = torchfun.relu(torch.bmm(inp, self.w3))    # third weights
        inp = torch.bmm(inp, self.w4).view(-1, J)       # fourth weights

        # add eta to inp[u][u]
        # eta_matrix = Variable(eta * torch.eye(J).type(torchten))
        # if args.cuda:
        # 	 eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        return torchfun.softmax(inp)

# =============================================================================
# training and testing routines
# =============================================================================
def train(net, optimizer, loss_normalizer, u):
    """
    Trains the Neural Network using IdProb4 on the training set.

    Args:
        net -- (IdProb5 instance)
        optimizer -- (torch.optim instance) of the Gradient-Descent function
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss
        u -- (Torch.Tensor) weights to be multiplied when calculating the loss
            function

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    loss, loop_time = 0, 0
    P_data = torch.zeros(num_train, J)

    for t in xrange(num_train):
        # build the input by appending trainR[t] to F_DIST
        inp = build_input(trainR[t])

        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
        loop_start = time.time()    # forgot to move this above the transfer,
        # did tests with this here (mistake), but ended not caring about how
        # much time this model took

        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax

        # calculate loss
        Pxt = torch.mv(P, trainX[t])
        P_data[t] = Pxt.data
        loss += (u[t] * (trainY[t] - Pxt)).pow(2).sum()

        loop_time += (time.time() - loop_start)

    # loss += args.lambda_L1 * torch.norm(net.w.data)
    start_outside = time.time()
    loss /= loss_normalizer

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = (time.time() - start_outside) + loop_time
    return (end_time, loss.data[0],
        torch.mean(P_data, dim=0).squeeze().cpu().numpy())

def test(net, loss_normalizer, u):
    """
    Tests the Neural Network using IdProb4 on the test set.

    Args:
        net -- (IdProb5 instance)
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss
        u -- (Torch.Tensor) weights to be multiplied when calculating the loss
            function

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    loss, loop_time = 0, 0
    P_data = torch.zeros(num_test, J)

    for t in xrange(num_test):
        # build the input by appending testR[t]
        inp = build_input(testR[t])

        if args.cuda:
            inp = inp.cuda()
        inp = Variable(inp)
        loop_start = time.time()    # forgot to move this above the transfer,
        # did tests with this here (mistake), but ended not caring about how
        # much time this model took

        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax

        # calculate loss
        Pxt = torch.mv(P, testX[t])
        P_data[t] = Pxt.data
        loss += (u[t] * (testY[t] - Pxt)).pow(2).sum()

        loop_time += (time.time() - loop_start)

    # loss += args.lambda_L1 * torch.norm(net.w.data)
    start_outside = time.time()
    loss /= loss_normalizer

    end_time = (time.time() - start_outside) + loop_time
    return (end_time, loss.data[0],
        torch.mean(P_data, dim=0).squeeze().cpu().numpy())

# =============================================================================
# utility functions for training and testing routines
# =============================================================================
def build_input(rt):
    """
    Builds and returns the input for the neural network. Joins F_DIST and R,
    expanding R to fit the dimension.

    Args:
        rt -- (Torch.Tensor) rewards vector to be appended to form the full
            dataset

    Returns:
        Torch.Tensor -- Input dataset for the neural network
    """
    if args.expand_R:
        # supplied rt is a matrix
        return torch.cat([F_DIST, rt.repeat(J, 1, 1)], dim=2)
    # else supplied rt is a vector
    return torch.cat([F_DIST, rt.repeat(J, 1)], dim=2)

# =============================================================================
# logs and plots
# =============================================================================
def save_plot(file_name, x, y, xlabel, ylabel, title):
    """
    Saves and (optionally) shows the loss plot of train and test periods.

    Args:
        file_name -- (str) name of the file for saving
        x -- (NumPy ndarray) data on the x-axis
        y -- (3d array/tuple) data on the y-axis. y[0] should be
            train results, y[1] should be test results obtained from the
            functions. y[-][k] should be the results after the k+1 epoch
            such that y[-][k][0] is the execution time and y[-][k][1] is the
            end loss. See the main area of the script on how this is built.
        xlabel -- (str) label for the x-axis
        ylabel -- (str) what else can it mean?
        title -- (str) title of the plot
    """
    # get the losses from data
    train_losses = [i for j in y[0] for i in j][1::2]
    test_losses = [i for j in y[1] for i in j][1::2]

    # plot details
    loss_fig = plt.figure(1)
    train_label, = plt.plot(x, train_losses, "r-", label="Train Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.title(title)

    # check if testing was enabled
    if args.should_test:
        test_label, = plt.plot(x, test_losses, "b-", label="Test Loss")
        plt.legend(handles=[train_label, test_label])
    else:
        plt.legend(handles=[train_label])

    # save and show
    loss_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_loss_plot:
        plt.show()
    plt.close()

def save_log(file_name, x, y, title):
    """
    Saves the log of train and test periods to a file.

    Args:
        file_name -- (str) name of the file
        x -- (NumPy ndarray) epoch data [1..number_of_epochs]
        y -- (3d array/tuple) same as that of save_plot()
        title -- (str) first line of the file
    """
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        f.write("J: %3d\t\tT: %3d\n-------------\n" % (J, T))
        for i in range(0, len(x), args.log_interval):
            # write data at log_intervals
            f.write("epoch = %d\t\ttrainloss = %.4f, traintime = %.4f" % (
                x[i], y[0][i][1], y[0][i][0]))
            if args.should_test:
                f.write("\t\ttestloss = %.4f, testtime = %.4f" % (
                    y[1][i][1], y[1][i][0]))
            f.write("\n")

def find_idx_of_nearest_el(array, value):
    """
    Helper function to plot_predicted_map(). Returns the index of the element in
    array closest to value

    Args:
        array -- (NumPy ndarray) array to be searched in
        value -- (float) closest number in array found for this number

    Returns:
        int -- index of the closest number to value in array
    """
    return (np.abs(array - value)).argmin()

def plot_predicted_map(file_name, lat_long, point_info, title, plot_offset=0.05):
    """
    Plots the a scatter plot of point_info on the map specified by the latitudes
    and longitudes and saves the plot to a image file

    Args:
        file_name -- (str) file name of the plot
        lat_long -- (NumPy ndarray) 2-d matrix of latitudes and longitudes of
            locations. The first column contains latitudes, and the second
            column contains longitudes.
        point_info -- (NumPy ndarray) Z values for all locations. The order of
            locations must be same as the order in lat_long
        title -- (str) title of the plot
        plot_offset -- (float) padding value for latitude and longitude in the
            plot (default=0.05)
    """
    # extract latitude and longitude
    lati = lat_long[:,0]
    longi = lat_long[:,1]
    # calculate plot dimensions - select between latitude/longitude based on
    # their span over earth. The greater span is the basis
    lo_min, lo_max = min(longi) - plot_offset, max(longi) + plot_offset
    la_min, la_max = min(lati) - plot_offset, max(lati) + plot_offset
    plot_width = max(lo_max - lo_min, la_max - la_min)
    lo_max = lo_min + plot_width
    la_max = la_min + plot_width

    # create the mesh for pcolormesh, see its documentation
    # retained step for convenience in testing
    # J+10 values needed on each side, this can lead to rectangular dots
    lo_range = np.linspace(lo_min, lo_max, num=J+10, retstep=True)
    la_range = np.linspace(la_min, la_max, num=J+10, retstep=True)
    lo, la = np.meshgrid(lo_range[0], la_range[0])

    z = np.zeros([J + 10, J + 10])
    for k in xrange(J):
        # for each location in latitude and longitude array, find the closest
        # value in the mesh, i.e., lati[k] in the mesh, longi[k] in the mesh
        lo_k_mesh = find_idx_of_nearest_el(lo[0], longi[k])
        la_k_mesh = find_idx_of_nearest_el(la[:, 0], lati[k])
        z[lo_k_mesh][la_k_mesh] = point_info[k] # assign Z value in the matrix

    map_fig = plt.figure(2)
    plt.pcolormesh(lo, la, z, cmap=plt.cm.get_cmap('Greys'), vmin=0.0, vmax=0.01)
    plt.axis([lo.min(), lo.max(), la.min(), la.max()])
    plt.colorbar()
    plt.title(title)
    map_fig.savefig(file_name, bbox_inches="tight", dpi=200)
    if not args.hide_map_plot:
        plt.show()
    plt.close()

# =============================================================================
# misc utility functions
# =============================================================================
def expand_R(rt, R_max=15):
    """
    Expands rt into a matrix with each rt[u] having R_max number of elements,
    where the first rt[u] elements are 1's and rest 0's. So if rt[u] is 7 and
    R_max is 15, rt[u] becomes [1 1 1 1 1 1 1 0 0 0 0 0 0 0 0].

    Args:
        rt -- (Torch.Tensor) vector of rewards
        R_max -- (int) Number of elements for expansion (default=15). When using
            orig data, R_max must be greater than 15. It's also the max reward in
            the rewards file

    Returns:
        Torch.Tensor -- Expanded R of size J x R_max
    """
    newrt = torchten(J, R_max)
    if args.cuda:
        newrt = newrt.cuda()
    for u in xrange(J):
        r = int(rt[u])
        newrt[u] = torch.cat([torch.ones(r), torch.zeros(R_max - r)], dim=0)
    return newrt

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    # READY!!
    read_set_data()
    net = IdProb5()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # SET!!
    # oops, realized that we skipped measuring the transfer time while
    # documenting after completing the project. Don't worry though, we don't
    # discuss the 4-layered model's runtimes in our final report.
    if args.cuda:
        # transfer net and tensors to the gpu
        net.cuda()
        trainX, trainY, trainR = trainX.cuda(), trainY.cuda(), trainR.cuda()
        testX, testY, testR = testX.cuda(), testY.cuda(), testR.cuda()
        F_DIST = F_DIST.cuda()
        u_train, u_test = u_train.cuda(), u_test.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    if args.expand_R:
        file_pre_gpu = "expandedR, " + file_pre_gpu

    # scalar + tensor not supported in pytorch v0.12.2
    # formula = (u(Y-mean(Y)))^2
    train_loss_normalizer = (torch.mv(torch.t(trainY \
        - torch.mean(trainY).expand_as(trainY)).data, u_train)).pow(2).sum()
    if args.should_test:
        test_loss_normalizer = (torch.mv(torch.t(testY \
            - torch.mean(testY).expand_as(testY)).data, u_test)).pow(2).sum()

    # GO!!
    train_time_loss, test_time_loss, total_time = [], [], 0.0
    for e in xrange(1, args.epochs + 1):
        # train
        train_res = train(net, optimizer, train_loss_normalizer, u_train)
        train_time_loss.append(train_res[0:2])  # the third element is not logged
        total_time += (train_res[0])

        # print results, some quirky arguments to print for nice console printing
        if e % 20 == 0:
            print("e= %2d, loss=%.8f" % (e, train_res[1]), end="")

        if args.should_test:
            # test
            test_res = test(net, test_loss_normalizer, u_test)
            test_time_loss.append(test_res[0:2])
            total_time += test_res[0]
            if e % 20 == 0:
                print(", testloss=%.8f\n" % (test_res[1]), end="")
        else:
            print("\n", end="")

        if e == args.epochs:
            # Network's final prediction
            y_pred = test_res[2] if args.should_test else train_res[2]

    # FINISH!!
    # log and plot the results: epoch vs loss

    # define file names
    if args.rand:
        file_pre = "randXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
        lat_long = ad.read_lat_long_from_Ffile(randF_file, J)
    else:
        file_pre = "origXYR_seed=%d, epochs=%d, " % (args.seed, args.epochs)
        lat_long = ad.read_lat_long_from_Ffile("../sensitive-avicaching/data/loc_feature_with_avicaching_combined.csv", J)
    log_name = "train=%3.0f%%, lr=%.3e, time=%.4f sec" % (
        args.train_percent * 100, args.lr, total_time)
    epoch_data = np.arange(1, args.epochs + 1)
    fname = "5layer_" + file_pre_gpu + file_pre + log_name
    # save amd plot data
    save_log(
        "./stats/find_weights/logs/" + fname + ".txt", epoch_data,
        [train_time_loss, test_time_loss], log_name)
    with open("./stats/find_weights/weights/" + fname + ".txt", "w") as f:
        # save w1
        w1 = net.w1.data.cpu().numpy()
        f.write('# w1 shape: {0}\n'.format(w1.shape))
        for data_slice in w1:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")

        # save w2
        w2 = net.w2.data.cpu().numpy()
        f.write('# w2 shape: {0}\n'.format(w2.shape))
        for data_slice in w2:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")

        # save w3
        w3 = net.w3.data.cpu().numpy()
        f.write('# w3 shape: {0}\n'.format(w3.shape))
        for data_slice in w3:
            f.write('# New slice\n')
            np.savetxt(f, data_slice, fmt="%.15f", delimiter=" ")

        # save w4
        w4 = net.w4.data.view(-1, numFeatures).cpu().numpy()
        f.write('# w4 shape: {0}\n'.format(w4.shape))
        np.savetxt(f, w4, fmt="%.15f", delimiter=" ")
    if not args.no_plots:
        # should plot
        save_plot(
            "./stats/find_weights/plots/" + fname + ".png", epoch_data,
            [train_time_loss, test_time_loss], "epoch", "loss", log_name)
        plot_predicted_map(
            "./stats/find_weights/map_plots/" + fname + ".png",
            lat_long, y_pred, log_name)

    print("---> " + fname + " DONE")
