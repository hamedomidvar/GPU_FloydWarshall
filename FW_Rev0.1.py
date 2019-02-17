## GPU-Based All Pairs Shortest Path (Floyd Warshall)

## Author: Hamed Omidvar
## Affilication: UC San Diego
## Aug. 2018


import numpy as np
import cupy as cp


## The Floyd-Warshall Algorithm
myInf = 1e6
def FW(W,n):
    ## Initially the distance between all pairs are set to infinity
    d = cp.ones([n,n])*myInf
    all_ones = cp.ones([n,n])
    for k in range(n):
        ## Calculating distance in iteration k
        dk = cp.log(cp.dot(cp.exp(W[:,k].reshape(n,1)),cp.exp(W[k,:].reshape(1,n))))
        a = cp.sign(d-dk)
        where_to = cp.where(a>0)
        ## Updating distances where necessary
        d[where_to] = dk[where_to]
    return d


## Example:

## Parameters
n = 4 ## Number of vertices


def make_W(n = 4):
    ## Making a random graph
    A = np.random.binomial(1,0.5,[n,n])
    A = np.triu(A,1) 
    A +=  np.transpose(A)
    A = cp.array(A)
    
    # Making random weights
    W = cp.array(np.random.exponential(scale=1,size=(n,n)))
    W = cp.multiply(A,W)
    W = cp.array(np.triu(W.get(),1)) 
    W +=  cp.transpose(W)
    W += (cp.ones([n,n])-A)*myInf
    cp.fill_diagonal(W,0)
    W = cp.array(W)
    return W



W = make_W()

print("Adjacency Matrix:", W)
print("Distances:", FW(W,n))