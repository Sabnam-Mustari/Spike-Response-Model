from matplotlib import pyplot as mp
from decimal import *
import numpy as np
import math
import functools
import pandas

wbc = pandas.read_csv('data.txt').as_matrix()
#iris = np.genfromtxt('iris.txt', delimiter=',')

wbc = wbc[:, 1:]  # First column(regarded as identity number) and lasr column (regarded as target value)has been dropped from dataset
train_Data = wbc[:200, :]  # 200 sample has been taken as training data
row, column = train_Data.shape
target_Data = wbc[:200, 9:10]  # last column represent target output

# ---------------------------------draw Gaussian distribution----------------------------------

A2 = 400
sig = 1 / (1.5 * 13)
x = np.linspace(0, .9, 110)
mu =[0, 0.083, .16, .24, .33, .41, .49, .58, .66, .74, .83, .91, .96, 1]
p=len(mu)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * A2

for i in range(0, p):
    mp.plot(gaussian(x, mu[i], sig))

arraylist = []
spikeop = np.zeros([10, 14])
spiketime = np.zeros([10, 14])

#--------------------- intersection point between line and Gaussian distribution-------------------------
'''
The intersection function return the crossing point among each straight line and gaussion distribution curve.
'''
def intersection(x_pos,row):
    for k in range(0, p):
        y=gaussian(x[x_pos], mu[k], sig)
        bd=Decimal(y)
        output = round(bd, 2)

        if(output>0):
           #arraylist.append (float("{0:.2f}".format(y)))
           spikeop[row][k] = float("{0:.1f}".format(y))
           spiketime[row][k] = 1
        else:
            #arraylist.append(0)
            spikeop[row][k] = 0
            spiketime[row][k] = 0

    return spiketime

# ---------------------------draw vertical line----------------------------
def drawline(x_pos):
    return mp.axvline(x_pos, ymin=0, ymax=1,linewidth = 2, color =  'black')
'''
trainingSet function is receiving a value for x from network that means this xth row now going to network as training set.
This function convert each value of xth into 14 spike time by calling intersection function.
we draw straight line for each value position of xth row dataset. The intersection function return the crossing point
among each straight line and gaussion distribution curve.
'''
def trainingSet(x):
    pattern = train_Data[x:x+1, :]       # for getting xth row input data to Tr_Set(Training Set)
    print('Dataset: ',pattern)
    row, column = pattern.shape
    for j in range(column):
        value= pattern[0][j]
        x_pos=(value *10)
        drawline(x_pos)
        intersection(x_pos,j)

    return (intersection(x_pos,j))

trainingSet(0)
#print('Spike Time:\n',spikeop)
#print(' Training Dataset:\n',spiketime)


def IH_weight(IN,HN):
    '''
    input layer has 9 neurons and hidden layer has 8 neurons,connection matrix[17 X 17]
    Matrix contain random weight in cell [1:9,10:17] and other position be 0
    '''
    toatl_neuron = IN + HN
    a = np.random.random((IN, HN)) # (9 X 8) Matrix
    b = np.zeros((HN, HN)) # (8 X 8) Matrix
    w = np.concatenate((a, b), axis=0) # (17 X 8) Matrix
    weight = np.zeros((toatl_neuron, IN))
    weight = np.concatenate((weight, w), axis=1) # (17 X 17) Matrix
    weight = (weight + weight.T) / 2  # symmetric matrix
    c, d = weight.shape
    #print(c, d)
    #print(weight)
    return weight
#IH_weight(9,8)

def HO_weight(HN,ON):
    '''
    Hidden layer has 8 neurons and Output layer has 2 neurons,connection matrix[10 X 10]
    Matrix contain random weight in cell [1:8,9:10] and other position be 0
    '''
    toatl_neuron = HN + ON
    a = np.random.random((HN,ON)) # (8 X 2) matrix
    b = np.zeros((ON, ON)) # (2 X 2) matrix
    w = np.concatenate((a, b), axis=0) # (10 X 2)
    weight = np.zeros((toatl_neuron, HN)) # (10 X 8)
    weight = np.concatenate((weight, w), axis=1) # (10 X 10) matrix
    weight = (weight + weight.T) / 2  # symmetric matrix
    c, d = weight.shape
    #print('A:',a,'B:', b)
    #print(weight)
    return weight
#HO_weight(8,2)
mp.show()