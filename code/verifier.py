import argparse
import torch
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

def deepZabstraction(out, neuron_i, lx_neuron_i, ux_neuron_i, x):

    s= ux_neuron_i/(ux_neuron_i-lx_neuron_i)  # slope (i.e. lambda value)
    for j in range(len(x)): #all coeff but biased are multiplied by slope
        out[j,neuron_i]= s*x[j,neuron_i]        
    out[0,neuron_i]= out[0,neuron_i] - s*lx_neuron_i/2            
    # add new row for the new epsilon coeff
    out= np.concatenate((out, np.zeros([1, len(out.T)])))
    out[len(out)-1,neuron_i]= - s*lx_neuron_i/2

    return out
    

def computeReLUabstraction(x):
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
    
#    out_prova_no_eps= np.zeros([len(x), len(x.T)]) #--> USED FOR TESTING set all eps to 0
    # 2) compute output ReLU abstraction  
    for i in range(len(x.T)):
        if lx[i]>0:
            out[:len(x),i]= x[:,i]
        elif ux[i]<=0:         
            1
           #out.append(np.zeros([len(x),1])) #no need because I initialize out to zero
        else:
            #crossing boundary case
            out= deepZabstraction(out, i, lx[i], ux[i], x)
            
#        out_prova_no_eps[0, i] = max(0, x[0,i])                        
    return out
    

def computeNNout(net, pxls, nn_type: str, eps: float):
    if nn_type== 'fc': # i.e. fully connected

        ''' FULLY CONNECTED NN   '''
        pixels= pxls.reshape(784)         
        a1= eps * np.ones(len(pixels)) # all coeff. for eps1
        p= np.asarray(pixels)
        #save each pixel and the coefficients of epsilon in same matrix
        out_ii = np.stack([p, a1], axis=0)
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
                out_ii= computeReLUabstraction(x_ii)

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
                vector_diff[j]= vector_diff[j]
            else:
                vector_diff[j]= -vector_diff[j]
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

    output_nn = computeNNout(net, inputs, nn_type, eps) 
    verified = is_verified (output_nn, true_label, eps)
    
    # return 1 if verified and 0 if not verified
    return verified


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')
        


if __name__ == '__main__':
    main()
