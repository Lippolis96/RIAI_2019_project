import argparse
import torch
from networks import FullyConnected, Conv
from networks_z import FullyConnectedZ, ConvZ, ReLUZ, ReLUConvZ

DEVICE = 'cpu'
INPUT_SIZE = 28

class lambdaConstraint():
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module, 'lambdas'):
            w = module.lambdas.data
            w = w.clamp(0.0, 1.0)
            module.lambdas.data = w


class VerificationMargin(torch.nn.Module):
    
    def __init__(self):
        super(VerificationMargin, self).__init__()
        
    def forward(self, outs, low_params, high_params, true_label):
        
        mask = torch.zeros_like(low_params)
        mask[:, true_label] = 1
        true_params = low_params[mask.bool()]
        diff_params = high_params + true_params.view(true_params.size(0), 1)
        
        mask = (diff_params<0).bool()
        ones = torch.ones_like(diff_params)
        ones[mask] = -ones[mask] 
        diff_params_abs = ones * diff_params

        diff_params_sum = torch.sum(diff_params_abs, axis = 0)
        diff_params_sum = diff_params_sum.unsqueeze(0)
        
        diff_outs = outs - outs[0, true_label]
        
        diff_tot = diff_outs + diff_params_sum
        
        diff_tot[0, true_label] =  float('-inf') 
        
        loss = torch.max(diff_tot)
        
        """
        Alternative loss to return:
            loss = torch.sum(diffs) - (true_high-true_low)
        This one already tries to push true low away from any other high,
        not only away from the other highest high.
        """
        
        # Here we return just the worst case
        return loss


class LambdaTuner():
    
    def __init__(self, net, inputs, low_params, high_params, true_label, lr, eps):
        self.net = net
        self.inputs = inputs
        self.low_params = low_params
        self.high_params = high_params
        self.true_label = true_label
        self.loss = VerificationMargin()
        
        self.constraints = lambdaConstraint()
        self.lambdas = []
        
        # Create list of optimizable params. Only this are updated (VERIFIED)
        for layer in self.net.layers:
            if type(layer) == ReLUZ or type(layer) == ReLUConvZ:
                self.lambdas.append(layer.lambdas)     
                
        self.optimizer = torch.optim.Adam(self.lambdas, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= 0.95)
    def _is_verified(self, loss):
        return loss.item() < 0

    def __call__(self):
        # Set training mode
        self.net.train()
        verified = False
        
        while not verified:

            self.optimizer.zero_grad()
            outs, low_params, high_params = self.net(self.inputs, self.low_params, self.high_params)
            loss = self.loss(outs, low_params, high_params, self.true_label)
            verified = self._is_verified(loss)
            if verified: break
            
            # Backward pass
            loss.backward()
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            # Constrain lambdas between [0,1]
            for layer in self.net.layers:
                if type(layer) == ReLUZ or type(layer) == ReLUConvZ:    
                    layer.apply(self.constraints)
            
        return verified
        
    
def analyze(net, inputs, low_params, high_params, true_label, lr, eps):
    """Function that analyzes the robustness of a z-model net to eps-attacks on 
    input inputs.
    
    Args:Module
        - net: z-model for bounds propagation
        - inputs: raw inputs before attacks
        - params: matrix containing eps disturbance on each input
        - true_label: true label for inputs
        
        ciao ivan sono frank e mi sono mandato tutto a casa xD
        cari saluti dalla Cina
    
    Returns True if robustness is verified, else False.
    """

    tuner = LambdaTuner(net, inputs, low_params, high_params, true_label, lr, eps)
    
    verified = tuner() # Boolean
    
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
        net_z = FullyConnectedZ(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
        net_z = FullyConnectedZ(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
        net_z = FullyConnectedZ(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
        net_z = FullyConnectedZ(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
        net_z = FullyConnectedZ(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
        net_z = ConvZ(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
        net_z = ConvZ(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
        net_z = ConvZ(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
        net_z = ConvZ(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
        net_z = ConvZ(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))
    net_z.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))
    
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    low_params = [min(eps, p) for i, p in enumerate(pixel_values)]
    high_params = [min(eps, 1-p) for i, p in enumerate(pixel_values)]

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if args.net.startswith('fc'):
        lr = 0.1
    else:
        lr = 0.01

    if eps == 0:
        print('verified')   
    else:
        if analyze(net_z, inputs, low_params, high_params, true_label, lr, eps):
            print('verified')
            
        else:
            print('not verified')


if __name__ == '__main__':
    main()
