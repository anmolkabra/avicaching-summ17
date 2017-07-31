#!/usr/bin/env python

# =============================================================================
# test_lp_time.py
# Author: Anmol Kabra -- github: @anmolkabra
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Only for logging LP runtime in the Pricing Problem's model.
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
import sys
import time
import argparse
import avicaching_data as ad
import lp
import torch, torch.nn as nn
import torch.nn.functional as torchfun
import torch.optim as optim
from torch.autograd import Variable

# =============================================================================
# training options
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding rewards")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=100, metavar="E",
    help="inputs the number of epochs to train for")
parser.add_argument("--threads", type=int, metavar="T",
    help="inputs the number of threads to run NN on")

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

torchten = torch.FloatTensor
J, T, totalR, numF = 100, 20, 1000, 10  # user must change it in script

# set the seeds
torch.manual_seed(1)
np.random.seed(seed=1)
if args.cuda:
    torch.cuda.manual_seed(1)

# shapes of datasets -- [] means expanded form:
# - X, Y: J
# - net.R: J [x J x 1]
# - F_DIST: J x J x numF
# - F_DIST_w1: J x J x numF
# - w1: J x J x numF
# - w2: J x numF [x 1]
# - w1_for_r: J x 1 x numF

# generate random data, no need to read from file
F_DIST_w1 = torch.randn(J, J, numF)
X, Y = torch.rand(J), torch.rand(J)
loss = 0
w1_for_r = torch.randn(J, 1, numF)
w2 = torch.randn(J, numF, 1)

F_DIST_w1 = Variable(F_DIST_w1, requires_grad=False)
w1_for_r = Variable(torchten(w1_for_r), requires_grad=False)
X = Variable(torchten(X), requires_grad=False)
w2 = Variable(torchten(w2), requires_grad=False)

###### 'ONLY LP' setting
# lp_A, lp_c = lp.build_A(J), lp.build_c(J)
# lp_time_log = []
# for e in xrange(args.epochs):
#     r_on_cpu = np.random.randn(J)
#     start_lp_time = time.time()
#    
#     # CONSTRAIN -- LP
#     # 1.0 is the sum constraint of rewards
#     # the first J outputs are the new rewards
#     lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
#     lp_time = time.time() - start_lp_time
#     print(lp_time)
#     lp_time_log.append([e, lp_time])
#     # net.R.data = torchten(lp_res.x[:J]).unsqueeze(dim=0)
# fname = "onlylp, epochs=%d, time=%.0f" % (args.epochs, time.time())
# np.savetxt("./stats/" + fname + ".txt", lp_time_log, fmt="%.6f", delimiter=",")
# sys.exit()
###########

# =============================================================================
# PriProb class -- copy from nnAvicaching_find_rewards.py
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
        2-tuple -- (Execution Time, End loss value)
    """
    global lp_A, lp_c, loss

    # BACKPROPAGATE
    start_backprop_time = time.time()
    
    optimizer.zero_grad()
    loss.backward()         # calculate grad
    optimizer.step()        # update rewards
    
    backprop_time = time.time() - start_backprop_time
    r_on_cpu = net.R.data.squeeze().cpu().numpy()   # transfer data for lp
    # don't need to log this transfer time

    start_lp_time = time.time()
    
    # CONSTRAIN -- LP
    # 1.0 is the sum constraint of rewards
    # the first J outputs are the new rewards
    lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
    lp_time = time.time() - start_lp_time
    # print(lp_time)
    net.R.data = torchten(lp_res.x[:J]).unsqueeze(dim=0)

    if args.cuda:
        # transfer data
        net.R.data = net.R.data.cuda()
    
    # FORWARD
    forward_time = go_forward(net)

    return (backprop_time + lp_time + forward_time, lp_time)

##### main script
net = PriProb()
if args.cuda:
    net.cuda()
    w1_for_r, w2, F_DIST_w1, X = w1_for_r.cuda(), w2.cuda(), F_DIST_w1.cuda(), X.cuda()

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(net.parameters(), 0.001)
lp_A, lp_c = lp.build_A(J), lp.build_c(J)

total_lp_time = 0
lp_time_log = []
total_time = go_forward(net)
print(torch.get_num_threads())

for e in xrange(1, args.epochs + 1):
    train_t = train(net, optimizer)
    curr_loss = loss.data[0]
    lp_time_log.append([e, train_t[1]])
    total_time += train_t[0]
    total_lp_time += train_t[1]
    if e % 20 == 0:
        print("epoch=%5d, loss=%.10f" % (e, curr_loss))

print(total_time, total_lp_time)
if args.cuda:
    file_pre = "gpu, "
else:
    file_pre = "cpu, "
fname = "threads=%d, epochs=%d, time=%.0f" % (args.threads, args.epochs, time.time())
np.savetxt("./stats/" + file_pre + fname + ".txt", lp_time_log, fmt="%.6f", delimiter=",")
