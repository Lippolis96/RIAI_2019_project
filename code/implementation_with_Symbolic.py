import argparse
import torch
import sympy     as sym
import numpy     as np
from networks    import FullyConnected, Conv

DEVICE = 'cpu'
INPUT_SIZE = 28

'''
TODO: check if we really need to carry epsilons al the time or we can just 
      keep coefficients
'''

# up  9.60206583831997
# down -9.50630096182213


def createZonotopeAbstraction (pixels):
    x= np.asarray(pixels)
    eps1 = sym.Symbol('eps1')
#    eps2 = sym.Symbol('eps2')
    zonotope= x + eps1 #+ eps2
    
    return zonotope



def computeBounds(x):
    lx= []
    ux= []
    for j in range(len(x)):
        a1= x[j].coeff('eps1')
        if a1>0:
            lx.append(x[j].subs([('eps1', -1)]))
            ux.append(x[j].subs([('eps1', +1)]))
        else:
            lx.append(x[j].subs([('eps1', +1)]))
            ux.append(x[j].subs([('eps1', -1)]))
         
    return lx, ux
                            
def createSymbolicVar(eps, layer: int, neuron: int):
    '''
    eps: contains all new variable created due to cross boundary in relu transformer
    e.g:    eps_new_ij is the var associated with layer i and neuron j
    '''
    eps_new = sym.Symbol('eps_new_'+str(layer)+'_'+str(neuron))
    eps.append(eps_new)
    
    return eps

    

def computeReLUabstraction(x, eps_relu: list, layer_i: int):
    '''
    x: input layer
    eps_relu: all new variables created up to now due to cross boundary of a ReLU
    '''
    
    # 1) check lower and upper bounds
    lx, ux= computeBounds(x)
    
    out= []
    # 2) compute output ReLU abstraction  
    for i in range(len(x)):
        if lx[i]>0:
            out.append(x[i])
        elif ux[i]<=0:
            out.append(0)
        else:
            #crossing boundary case
            s= ux[i]/(ux[i]-lx[i])  # slope (i.e. lambda value)
            eps_relu= createSymbolicVar(eps_relu, layer_i, i) # note that i is the neuron 
            eps_i= eps_relu[len(eps_relu)-1] # take symbolic variable just added
            out.append(s*x[i]-eps_i*s*lx[i]/2-s*lx[i]/2)
            
            
    return out
            
            
                        
def analyze(net, inputs, eps, true_label):
    
    epsReLU= [] # create empty element to store possible new dummy variable in
                # ReLU abstraction
                
                
    # create zonotope abstraction
    pixels= inputs.reshape(784) 
    zonotopeAbstraction= createZonotopeAbstraction(pixels)
    
    ''' FULLY CONNECTED NN   '''

    #run zonotope through nn 
    out_i = zonotopeAbstraction
    for i in range(2,len(net.layers)): # start from 2 because layer 1 is 
                                       # normalization and layer 2 is flattening
        x_i= out_i                               
        if type(net.layers[i])==torch.nn.modules.linear.Linear:
            ''' zonotope affine transformer '''
            w_i= np.asarray(net.layers[i].weight.detach())   
            b_i= np.asarray(net.layers[i].bias.detach())
            
            out_i= np.dot(w_i, x_i) + b_i
        elif type(net.layers[i])==torch.nn.modules.activation.ReLU:
            ''' zonotope ReLU transformer '''
            out_i= computeReLUabstraction(x_i, epsReLU, i)
            
            
    out_n= out_i
        
        
        #    '''     CONVOLUTIONAL NN       '''
#    elif type(net.layers[1])==torch.nn.modules.conv.Conv2d:
    



    
    # return 1 if verified and 0 if not verified
    return out_n





'''                 MAIN               '''


net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
net.load_state_dict(torch.load('../mnist_nets/%s.pt' % 'fc2', map_location=torch.device(DEVICE)))

spec= '../test_cases/fc2/img0_0.05500.txt'
with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])


inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
outs = net(inputs)
pred_label = outs.max(dim=1)[1].item()
assert pred_label == true_label


if 1: #analyze(net, inputs, eps, true_label):
    print('verified')
else:
    print('not verified')



#output1= analyze(net, inputs, eps, true_label)
    


