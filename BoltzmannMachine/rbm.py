# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):     # self: used access the local variables corresponding to this classs among all functions defined, nv,nh: number of visible and hidden nodes
        self.W = torch.randn(nh,nv) # randn is a random dustribution function with mean 9 and variance 1 used to initilize weights
        self.a = torch.randn(1,nh)  # initialization of bias a
                                    # a is bias for the probability of hidden node is 1 given the visible node: p(h=1/v)
                                    # 1 correspinds to batch number/size and nh corresponds to bias
                                    # it is 2D vector because pytorch always accepts 2D
                                    # Rememeber there are 1 bias for each hidden nodes and since we have nh numbers of hidden nodes so we 
                                    # we had to create an vector of nh, we could have simply created only 1D vector of nh by passing only
                                    # nh as the the argument but pytorch always needs to 2d vector so the format is (1,nh)
        self.b = torch.randn(1,nv)  # initialization of bias b
                                    # b is bias for the probability of visible node given the hidden node: p(v/h)                                    

    """# Next function Sampling the hidden nodes according to p(h=1/v) which is nothing but a sigmoid function
    # This sampling is required because during the training of data we need to perform 
    # log liklihood of gradient using Gibbs sampling
    
    #self: used access the local variables corresponding to this classs among all functions defined,
    #x: corresponds to visible neurons-v used in p(h=1/v)
    #p(h=1/v)->sigmoid activation function applied to product of weights and visible neurons plus bias corresponds to hidden nodes. i.e p(h=1/v) = w*x + a
    #torch.mm() -> matrix multiplication
    #self.a.expand_as(wx)-> is used to expand bias 'a' as matrix multiplication result 'wx' to each line of mini batch, where mini batch is a batch of only 1 vector created above ->self.a = torch.randn(1,nh)
    #generaly a batch contains many lines of input row like we have seen in other tutorials ex: 32, but in this perticular example there is only 1 input row inside each batch, so it is called mini batch
    #activation = wx + self.a.expand_as(wx) -> linear equation/combination: where x = visible neuons, coefficient = weights(w), bias = self.a.expand_as(wx)
    #This activation will have the probability of hidden node that will be activated according to the value of visible node, since this is an sigmoid activation function here we will try to find what is 
    #the probability of hidden node reaching clos to value 1(i.e. activated).
    #We are predicting a binary outcome i.e a movie is liked or not. So we are using Bernoulli RBM samples for p(h=1/v)
    #Ex: if the ith elemnt of the hidden node vector (i.e (1,nh)) is p_h_given_v then it's corresponding bernoullis distribution is given by torch.bernoulli(p_h_given_v)"""
    def sample_h(self, x): # returns sampling distribution for p(h=1/v), x: coressponds to the value of visible node
        wx = torch.mm(x, self.W.t()) # Transpose is not required because weight metrix W is for p(v/h) which is opposite that of x
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y): # returns sampling distribution for p(v/h), y: coressponds to the value of hidden node
        wy = torch.mm(y, self.W) # Transpose is not required because weight metrix W is for p(v/h) which same as y
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)  
    
        """v0 = 0th visibility node after 0 iteration or ist time
        vk = kth visibility node after k iteration
        ph0 = p(h0 = 1/v0), probability of hideen node h0 is 1 given the visibility node v0 at first iteration
        phk = p(hk = 1/vk), probability of hideen node hk is 1 given the visibility node vk after K iteration
        h0= 0th hidden node
        hk= kth hidden node"""
    def train(self, v0, vk, ph0, phk): # Implementation of contrastive divergence, equation can be found in pdf given along with dataset
        #self.W += torch.mm((v0.t(), ph0) - torch.mm(vk.t(),phk))
        self.W += torch.mm(ph0, v0) - torch.mm(phk, vk)  #The order of tensors multiplied when the 'mm' (tensor product) method is applied must be inverted so that the products can be performed. The transposing of v0 and vk must also be eliminated (not required).
        self.b += torch.sum((v0 - vk),0)   #0 is added to make a Tensor of 2D
        self.a += torch.sum((ph0 - phk),0) #0 is added to make a Tensor of 2D

nv = len(training_set[0])  # number of elements in first line of training_set tensor. This is exactly same as number of movies of number of visisble nodes.       
nh = 100
batch_size = 100
rbm = RBM(nv,nh)    
        
# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.0
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.0
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0.0
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # V = prediction，the training set will be use to activate the hidden neuron to get the output
                                        # i.e we are using the inputs of the training set to activate the neurons of the RBM to get the predicted ratings of the test set

    vt = training_set[id_user:id_user+1] # vt = target, contains the original rating of test set
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.0
print('test loss: '+str(test_loss/s))
    