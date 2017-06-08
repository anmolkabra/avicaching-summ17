from __future__ import print_function
import argparse, time, csv
import numpy as np
# import torch packages
import torch
import torch.nn as nn
from torch.autograd import Variable

# training specs
parser = argparse.ArgumentParser(description="NN for Avicaching model")
parser.add_argument("--batch-size", type=int, default=64, metavar="B",
    help="inputs batch size for training (default=64)")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
    help="inputs learning rate of the network (default=0.01)")
parser.add_argument("-momentum", type=float, default=0.5, metavar="M",
    help="inputs SGD momentum (default=0.5)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epoch", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
parser.add_argument("--locations", type=int, default=116, metavar="L",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=10, metavar="T",
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
torchten = torch.FloatTensor
X, Y, R, F, DIST = [], [], [], [], []


def read_XYR_file():
    global X, Y, R, J
    with open("./density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", 
        "r") as xyrfile:
        for idx, line in enumerate(xyrfile):
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
            line_vec = np.array(map(float, line.split(",")[:-1]))  # ignore last col
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
        for idx, line in enumerate(distfile):
            line_vec = np.array(map(float, line.split()))
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

# change the input data into pytorch nn variables
X, Y, R = torchten(X), torchten(Y), torchten(R)
F, DIST = torchten(F), torchten(DIST) # FloatTensors
print(X)
print(F)
print(DIST)

# objective
# Minimize the cost function 
# C(w) = \sum_{j,t} (u_{j,t}(y - P(f,r;w)x))^2 + lam * |w| by adjusting w

# Thus find P[J][J], where 
# p_{u,v} is a softmax function with weighted input of the form
# w {dot} phi(f_{u,v}, r_u) + eta {dot} I (I is the identity matrix)

# -------------------------------
# input to NN:
#   phi(f, r) for each p_{u,v}
# output from NN:
#   softmax(w {dot} phi(f, r) + eta {dot} I)
# loss function
#   target labels -> y
#   output labels -> Px

# NN class
class Net(nn.Module):
    def __init__(self):
        """
        Initialiizes the Conv. Neural Network
        """
        super(Net, self).__init__()
        # 2 hidden layers
        # size of NN: JxJ x 100x100 x 
        #
        self.conv1 = nn.Conv2d()

    def forward(self, input):
        """
        The Input is propagated forward in the network
        """
        # go through the layers
        # ...
        # output the softmax
        return torch.nn.functional.log_softmax(input)

def train(net, epoch, optimizer):
    """
    Trains the network
    """
    net.train() # comment out if dropout or batchnorm module not used
    for batch_idx, (data, target) in enumerate(train_data):
        # for each batch
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        # go forward in the network, process to get outputs
        output = net(data)
        
        # backpropagate
        optimizer.zero_grad()
        loss = torch.nn.MSELoss(output, target)     # calculate loss
        loss.backward()                 # backpropagate loss
        optimizer.step()                # update the weights

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



# if __name__ == "__main__":
#     net = Net()
#     if args.cuda:
#         # move the network to the GPU, if CUDA supported
#         net.cuda()

#     # using SGD as the optimizer function
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#     for epoch in xrange(1, args.epochs + 1):
#         train(net, epoch, optimizer)
#         # test(net, epoch)