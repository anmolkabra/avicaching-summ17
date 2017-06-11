from __future__ import print_function
import argparse, time, csv
import numpy as np
# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as torchfun
from torch.autograd import Variable
import torch.optim as optim

# training specs
parser = argparse.ArgumentParser(description="NN for Avicaching model")
parser.add_argument("--batch-size", type=int, default=64, metavar="B",
    help="inputs batch size for training (default=64)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
    help="inputs learning rate of the network (default=0.01)")
parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
    help="inputs SGD momentum (default=0.5)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
parser.add_argument("--locations", type=int, default=116, metavar="L",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=173, metavar="T",
    help="inputs total time of data collection; number of weeks (default=10)")
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="inputs parameter eta in the model (default=10.0)")
parser.add_argument("--lambda-L1", type=float, default=10.0, metavar="LAM",
    help="inputs the L1 regularizing coefficient") 

args = parser.parse_args()

# assigning cuda check to a single variable
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

# parameters and data
J, T, eta, l = args.locations, args.time, args.eta, args.lambda_L1
torchten = torch.DoubleTensor
X, Y, R, F, DIST = [], [], [], [], []


def read_XYR_file():
    global X, Y, R, J, T
    with open("./density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", 
        "r") as xyrfile:
        for idx, line in zip(xrange(T * 3), xyrfile):
            line_vec = np.array(map(float, line.split()[:J]))   # only J cols
            if idx == 0:
                # X init
                X = line_vec
            elif idx == 1:
                # Y init
                Y = line_vec
            elif idx == 2:
                # R init
                R = line_vec
            else:
                rem = idx % 3
                if rem == 0:
                    # append X information
                    X = np.vstack([X, line_vec])
                elif rem == 1:
                    # append Y information
                    Y = np.vstack([Y, line_vec])
                else:
                    # append R information
                    R = np.vstack([R, line_vec])
    
def read_Fu_file():
    global F
    with open("./loc_feature_with_avicaching_combined.csv", "r") as fufile:
        next(fufile)        # skip header row of fufile
        for idx, line in enumerate(fufile):
            line_vec = np.array(map(float, line.split(",")[:-3]))  # ignore last 3 cols
            if idx == 0:
                # F init
                F = line_vec
            else:
                # append F info
                F = np.vstack([F, line_vec])

def read_dist_file():
    global DIST
    with open("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt",
        "r") as distfile:
        for idx, line in zip(xrange(J), distfile):
            line_vec = np.array(map(float, line.split()))[:J]
            if idx == 0:
                # DIST init
                DIST = line_vec
            else:
                # append DIST info
                DIST = np.vstack([DIST, line_vec])

read_XYR_file()
read_Fu_file()
read_dist_file()

# input data
# y_{j,t}: y[j] is the probability of a birder (normalized) visiting 
#   location j at time t after rewards have been applied
# x_{j,t}: x[j] is the known number of visits (normalized) to location j 
#   at time t by a birder.
# r_t: reward vector of length J, where r[j] signifies the reward assigned
#   to location j at time t
# f_t: feature vector of length J, where f[j] signifies the reward assigned
#   to location j at time t
# u_{j,t}: adjusted weights

# process data for the NN
numFeatures = len(F[0]) + 1     # distance and reward also included
NN_in = np.empty([J, J, numFeatures], dtype=float)

# combine F and DIST
def combine_DIST_F():
    global F, DIST, NN_in
    for v in xrange(len(DIST)):
        for u in xrange(len(DIST[0])):
            NN_in[v][u][0] = DIST[v][u]
            NN_in[v][u][1:] = F[u]

combine_DIST_F()

# change the input data into pytorch nn variables
X, Y, R = torchten(X), torchten(Y), torchten(R)
F, DIST = torchten(F), torchten(DIST) # FloatTensors
NN_in = torchten(NN_in)
# print(X)
# print(F)
# print(R)
# print(DIST)
# print(NN_in)

numFeatures += 1    # adding R layer to NN_in later
# objective
# Minimize the cost function 
# C(w) = \sum_{j,t} (u_{j,t}(y - P(f,r;w)x))^2 + lam * |w| by adjusting w

# Thus find P[J][J], where 
# p_{u,v} is a softmax function with weighted input of the form
# w {dot} phi(f_{u,v}, r_u) + eta {dot} I (I is the identity matrix)

# -------------------------------
# input to NN:
#   for each v:
#       cat((cat((f_u1, r_u1), dim=0), cat((f_u2, r_u2), dim=0), ... , cat((f_un, r_un), dim=0)), dim=1)
#       total (numFeatures+1) x J mat, where mat[end][:] is the reward vector 
#           and mat[i][j] is the ith feature (i != end) at jth location, compared to vth location
# output from NN:
#   for each v:
#       [p_u1v, p_u2v, ..., p_uJv] = [softmax(w {dot} mat + eta {dot} I)]
# loss function
#   target labels -> y
#   output labels -> Px
# -------------------------------

# NN class
class Net(nn.Module):
    def __init__(self, J, numFeatures):
        """
        Initialiizes the Neural Network, takes in a 2D Tensor of size 
        (J * numFeatures+1) and returns a 1D vector of size J
        """
        super(Net, self).__init__()
        # 2 hidden layers 
        self.fc1 = nn.Linear(numFeatures, 1, bias=False).double()

    def forward(self, inp):
        """
        The Input is propagated forward in the network
        in -> conv1 -> pool -> relu -> conv2 -> dropout -> pool -> relu -> 
        linear -> relu -> linear -> relu -> linear -> softmax -> out
        """
        # go through the layers
        inp = self.fc1(inp)
        #print("after fc1, ", inp)
        # output the softmax
        #print("after softmax, ", torch.nn.functional.softmax(inp.t()))
        return torchfun.softmax(inp.t())

def train1(net, epoch, criterion, optimizer):
    """
    Trains the network
    """
    global X, Y, R, NN_in, J, numFeatures, T
    loss = 0
    P = Variable(torchten(J, J), requires_grad=False)
    X, Y = Variable(X, requires_grad=False), Variable(Y, requires_grad=False)
    #NN_in_exten = Variable(NN_in)
    for e in xrange(epoch):
        print(e)
        for t in xrange(T):        # [0, 173)
            R_extended = R[t].repeat(J, 1)
            NN_in_exten = torch.cat([NN_in, R_extended], dim=2).double()   # final NN_in preprocessing
            NN_in_exten = Variable(NN_in_exten)
            #print("NN_in_extend", NN_in_exten)
            for v in xrange(J):
                P[v] = net(NN_in_exten[v])
                #print("P[%d] = " % (v), P[v])

            P = P.t().clone()
            print(t, P)
            Pxt = torch.mv(P.clone(), X[t])
            loss += (Y[t] - Pxt).pow(2).sum()

        optimizer.zero_grad()
        loss.backward(retain_variables=True)
        optimizer.step()
        print(loss.data[0])

    # for batch_idx, (data, target) in enumerate(train_data):
    #     # for each batch
        
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
        
    #     # go forward in the network, process to get outputs
    #     # the output is a vector of u_i for a single v with softmax applied
    #     output = net(data)
    #     # backpropagate
    #     optimizer.zero_grad()
    #     loss = torch.nn.MSELoss(output, target)     # calculate loss
    #     loss.backward()                 # backpropagate loss
    #     optimizer.step()                # update the weights

def train2(net, epoch, criterion, optimizer):
    """
    Trains the network
    """
    global X, Y, R, NN_in, J, numFeatures, T
    X, Y = Variable(X, requires_grad=False), Variable(Y, requires_grad=False)
    w_t = Variable(torch.randn(J, numFeatures, 1).type(torchten))
    P = Variable(torchten(J, J), requires_grad=True)
    net = nn.Softmax()
    for e in xrange(epoch):
        print(e)
        loss = 0
        
        optimizer.zero_grad()

        for t in xrange(T):
            #print(R[t])
            R_extended = R[t].repeat(J, 1)
            NN_in_exten = torch.cat([NN_in, R_extended], dim=2)     # final NN_in_processing
            NN_in_exten = Variable(NN_in_exten, requires_grad=True)
            w_phi = torch.bmm(NN_in_exten, w_t).view(-1, J)         # view to convert to 2D tensor
            
            P = net(w_phi).t().clone()
            #print("P, ", P)
            Pxt = torch.mv(P.clone(), X[t])
            #print("Y[t] - Pxt", Y[t] - Pxt)
            loss += (Y[t] - Pxt).pow(2).sum()
            #print(w_phi)
            #print(torchfun.softmax(w_phi))
        
        print(loss.data[0])
        loss.backward()
        optimizer.step()
        #print(loss.data[0])
            



    # loss = 0
    # P = Variable(torchten(J, J), requires_grad=False)
    # X, Y = Variable(X, requires_grad=False), Variable(Y, requires_grad=False)
    # #NN_in_exten = Variable(NN_in)
    # for e in xrange(epoch):
    #     print(e)
    #     for t in xrange(T):        # [0, 173)
    #         R_extended = R[t].repeat(J, 1)
    #         NN_in_exten = torch.cat([NN_in, R_extended], dim=2).double()   # final NN_in preprocessing
    #         NN_in_exten = Variable(NN_in_exten)
    #         #print("NN_in_exten", NN_in_exten)
    #         for v in xrange(J):
    #             P[v] = net(NN_in_exten[v])
    #             #print("P[%d] = " % (v), P[v])

    #         P = P.t().clone()
    #         print(t, P)
    #         Pxt = torch.mv(P.clone(), X[t])
    #         loss += (Y[t] - Pxt).pow(2).sum()

    #     optimizer.zero_grad()
    #     loss.backward(retain_variables=True)
    #     optimizer.step()
    #     print(loss.data[0])

    # for batch_idx, (data, target) in enumerate(train_data):
    #     # for each batch
        
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
        
    #     # go forward in the network, process to get outputs
    #     # the output is a vector of u_i for a single v with softmax applied
    #     output = net(data)
    #     # backpropagate
    #     optimizer.zero_grad()
    #     loss = torch.nn.MSELoss(output, target)     # calculate loss
    #     loss.backward()                 # backpropagate loss
    #     optimizer.step()                # update the weights

def test(net, epoch):
    """
    Test the network, unsure if needed
    """
    net.eval()  # comment out if dropout or batchnorm module not used
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size



if __name__ == "__main__":
    net = Net(J, numFeatures)
    if args.cuda:
        # move the network to the GPU, if CUDA supported
        net.cuda()

    # using SGD as the optimizer function
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # for epoch in xrange(1, args.epochs + 1):
    # train1(net, args.epochs + 1, criterion, optimizer)
    train2(net, args.epochs + 1, criterion, optimizer)
#         # test(net, epoch)