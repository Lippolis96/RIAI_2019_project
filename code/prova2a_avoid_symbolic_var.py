#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:05:28 2019

@author: ivanlippolis
"""

import argparse
import torch
import sympy     as sym
import numpy     as np
from networks    import FullyConnected, Conv

DEVICE = 'cpu'
INPUT_SIZE = 28


def computeBounds(params):
    lx= []
    ux= []
    x= params[0,:]
    for j in range(len(x)):
        ux.append(x[j])
        lx.append(x[j])
        for i in range (1,len(params)): # start from 1 because we already sum pxl value 
            if params[i,j]>=0:
                ux[j]= ux[j] + params[i,j]
                lx[j]= lx[j] - params[i,j]
            else:
                ux[j]= ux[j] - params[i,j]
                lx[j]= lx[j] + params[i,j]
                 
    return lx, ux

def deepZabstraction(out, neuron_i, lx_neuron_i, ux_neuron_i, x, k):
    dEps= k*ux_neuron_i/100
    s= (ux_neuron_i + dEps)/(ux_neuron_i-lx_neuron_i)  # slope (i.e. lambda value)
    for j in range(len(x)): #all coeff but biased are multiplied by slope
        out[j,neuron_i]= s*x[j,neuron_i]        
    out[0,neuron_i]= out[0,neuron_i] - s*lx_neuron_i/2            
    # add new row for the new epsilon coeff
    out= np.concatenate((out, np.zeros([1, len(out.T)])))
    out[len(out)-1,neuron_i]= - s*lx_neuron_i/2

    return out
    

def computeReLUabstraction(x, k):
    '''
    x: matrix defined as [a1; x;  a2; new_eps; bias]
        a1: coeff. epsilon1
        a2: coeff. epsilon2
        new_eps: vector of coeff due to new epsilon introduced by Relu
        
    layer_i: actual layer in which we are computing Relu abstraction
    '''
    
    # 1) check lower and upper bounds
    lx, ux= computeBounds(x)
    out= np.zeros([len(x), len(x.T)])
    
    out_prova_no_eps= np.zeros([len(x), len(x.T)]) #--> USED FOR TESTING set all eps to 0
    # 2) compute output ReLU abstraction  
    for i in range(len(x.T)):
        if lx[i]>0:
            out[:len(x),i]= x[:,i]
        elif ux[i]<=0:         
            1
           #out.append(np.zeros([len(x),1])) #no need because I initialize out to zero
        else:
            #crossing boundary case
            out= deepZabstraction(out, i, lx[i], ux[i], x, k)
            
        out_prova_no_eps[0, i] = max(0, x[0,i])                        
    return out_prova_no_eps
    

def computeNNout(net, pxls, nn_type: str, k: int):
    if nn_type== 'fc': # i.e. fully connected

        ''' FULLY CONNECTED NN   '''
        pixels= pxls.reshape(784)         
        a1= np.ones(len(pixels)) # all coeff. for eps1
        p= np.asarray(pixels)
        a2= np.ones(len(pixels)) # all coeff. for eps2        
        #save each pixel and the coefficients of epsilon in same matrix
        out_ii = np.stack([p, a1, a2], axis=0)
        #run zonotope params through nn 
        for i in range(2,len(net.layers)): # start from 2 because layer 1 is 
                                           # normalization and layer 2 is flattening
            x_ii= out_ii
            if type(net.layers[i])==torch.nn.modules.linear.Linear:
                ''' zonotope affine transformer '''
                w_i= np.asarray(net.layers[i].weight.detach())   
                b_i= np.asarray(net.layers[i].bias.detach())                
                out= []
                for j in range(len(x_ii)):
                    out.append(np.dot(w_i, x_ii[j,:]))                
                out[0]=out[0]+b_i # add bias to pixel value
                out_ii= np.asarray(out)
                
            elif type(net.layers[i])==torch.nn.modules.activation.ReLU:
                ''' zonotope ReLU transformer '''
                out_ii= computeReLUabstraction(x_ii, k)

        out_nn= out_ii
    elif nn_type== 'cnn': # i.e. convolutional nn
        1
    ''' CONVOLUTIONAL NN   '''

    '''
    TODO
    '''        
    
    return out_nn
    
def is_verified(coeff, true_label, eps):
        
    true_label_coeff= coeff[:, true_label]
    rest_coeff= np.delete(coeff, true_label, 1)
    verified= 0    
    count= 0
    for i in range(len(rest_coeff.T)):
        vector_diff= true_label_coeff - rest_coeff[:,i]
        for j in range(1, len(rest_coeff)):
            if vector_diff[j]<0:
                vector_diff[j]= eps*vector_diff[j]
            else:
                vector_diff[j]= -eps*vector_diff[j]
        worst_case_i= np.sum(vector_diff)
        if worst_case_i >=0:
            count = count +1
        
            
    if count == len(rest_coeff.T):
        verified= 1
        
    return verified

    
                     
def analyze(net, inputs, eps, true_label):
    
    if type(net.layers[1])==torch.nn.modules.flatten.Flatten:            
        nn_type= 'fc'
    elif type(net.layers[1])==torch.nn.modules.conv.Conv2d:
        nn_type= 'cnn'
        
    verified= 0
    k=0
    k_lim= 10
    while (verified==0 and k<k_lim):
        output_nn = computeNNout(net, inputs, nn_type, k) 
        verified = is_verified (output_nn, true_label, eps)
        k= k+1
    
    # return 1 if verified and 0 if not verified
    
    if verified:
        print('verified for k = %d', k-1)
    
    return verified





'''                 MAIN               '''




net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
net.load_state_dict(torch.load('../mnist_nets/%s.pt' % 'fc1', map_location=torch.device(DEVICE)))


spec= '../test_cases/fc1/img1_0.07500.txt'
with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])


inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
outs = net(inputs)
pred_label = outs.max(dim=1)[1].item()
assert pred_label == true_label

#output_nn= computeNNout(net, inputs, 'fc', 0)
if analyze(net, inputs, eps, true_label):
    print('verified')
else:
    print('not verified')




