import torch
import sympy     as sym
import numpy     as np
from networks    import FullyConnected, Conv

DEVICE = 'cpu'
INPUT_SIZE = 28


'''                 MAIN               '''




net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
net.load_state_dict(torch.load('../mnist_nets/%s.pt' % 'fc1', map_location=torch.device(DEVICE)))

spec= '../test_cases/fc1/img0_0.06000.txt'
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

pixels= inputs.reshape(784) 
out_i= np.asarray(pixels)
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
        out_i= []
        for i in range(len(x_i)):
            out_i.append( max(0, x_i[i]))
        
        
out_n= out_i
