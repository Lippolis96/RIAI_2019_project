import torch
import torch.nn as nn

DEVICE = 'cpu'
INPUT_SIZE = 28


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma

def _abs_tensor(tensor):
        mask = (tensor<0).bool()
        ones = torch.ones_like(tensor)
        ones[mask] = -ones[mask] 
        tensor_abs = ones * tensor
        
        return tensor_abs
    
def _flip_low_high(low_params, high_params):
    
    mask_low_neg = (low_params<0).bool()
    low_params_flip = torch.zeros_like(low_params)
    high_params_flip = torch.zeros_like(high_params)
    low_params_flip[~mask_low_neg], high_params_flip[~mask_low_neg] = low_params[~mask_low_neg], high_params[~mask_low_neg]
    low_params_flip[mask_low_neg], high_params_flip[mask_low_neg] = high_params[mask_low_neg], low_params[mask_low_neg]
    
    return low_params_flip, high_params_flip    

class ReLUZ(nn.Module):
    
    def __init__(self, device, input_size):
        super(ReLUZ, self).__init__()
        # Initializing lambdas
        self.lambdas = torch.ones(input_size, requires_grad = True, device=device)
        self.initialized = False  # This flag is set to True after the first 
                                  # forward propagation, which allows to 
                                  # initialize lambda as DeepZ 
                         
    def _computeBounds(self, x, low_params, high_params):
        
        low_params_sum = torch.sum(low_params, axis = 0)
        high_params_sum = torch.sum(high_params, axis = 0)
        low_params_sum.unsqueeze(0)
        high_params_sum.unsqueeze(0)
        low = x - low_params_sum
        high = x + high_params_sum
        return low, high
    
    def _zonotope(self, low, high, params):
        new_params = self.lambdas*params
        mask_to_initial = torch.cat(new_params.size(0)*[low>0])
        new_params[mask_to_initial] = params[mask_to_initial]
        mask_to_zero = torch.cat(new_params.size(0)*[high<=0])
        new_params[mask_to_zero] = 0
        
        # Add params if and only if any cross boundary exists
        if ((low<0)*(high>0)).any().item():
            # Create new line with new parameters. For now they are all in the 
            # same channel.
            new_fused_line = torch.zeros((1, params.size(1)))
            new_fused_line = (-0.5*self.lambdas*low)
            # Compute params in special case 0 <= lambda < high/(high-low)
            mask_lambdas = (self.lambdas) < (high/(high-low))
            new_fused_line[mask_lambdas] = new_fused_line[mask_lambdas] - (0.5*(-high+self.lambdas*(high-low)))[mask_lambdas]
            # Concatenate as many new_fused_lines as many cross boundaries we have
            new_fused_line = torch.cat([new_fused_line for i in range(torch.sum((low<0)*(high>0)).item())])
            # Create a mask to put to preserve for each channel of new_fused_line only the corresponding new param
            zeros = torch.zeros_like(new_fused_line)
            indexes =  [tuple(torch.nonzero((low<0)*(high>0))[i].numpy()) for i in range(torch.sum((low<0)*(high>0)).item())]
            indexes =  [(idx[0]+i, idx[1]) for i, idx in enumerate(indexes)]
            for idx in indexes:
                zeros[idx] = 1
            new_fused_line = new_fused_line*zeros
            # Cat the so created new_fused_line with previously existing params
            new_params = torch.cat((new_params, new_fused_line))
        
        return new_params
        
    def forward(self, x, low_params, high_params):
        low, high = self._computeBounds(x, low_params, high_params) 
        assert (low<=high).all()  # Sanity check
        if not self.initialized:            
            self.lambdas.data = high/(high-low+1e-5) # 1e-5 to avoid division by zero
            self.initialized = True
        
        z = self.lambdas*(x-0.5*low)
        bias = -0.5*(-high+self.lambdas*(high-low))
        z[(self.lambdas<(high/(high-low)))] = (z + bias)[(self.lambdas<(high/(high-low)))]
        z[low>0] = x[low>0]
        z[high<=0] = 0
        
        new_low_params = self._zonotope(low, high, low_params)
        new_high_params = self._zonotope(low, high, high_params)
        
        return z, new_low_params, new_high_params 


class FullyConnectedZ(nn.Module):

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnectedZ, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZ(device, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def _params_to_flattened_tensor(self, params):
        params_tensor = None
        for i, p in enumerate(params):  
            line = [0 for _ in params]
            line[i] = p
            line = torch.FloatTensor(line).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
            line = self.layers[1](line)/0.3081  # Flatten + Normalize
            # Normalization on a delta means just dividing by the variance. 
            # Mean should not be substracted
            
            if params_tensor is None:
                params_tensor = line
            else:
                params_tensor = torch.cat((params_tensor, line), axis=0)
                    
        params = params_tensor
        return params


    def forward(self, x, low_params, high_params):
        
        x = self.layers[0](x)  # Normalize
        x = self.layers[1](x)  # Flatten
        low_params = self._params_to_flattened_tensor(low_params)
        high_params = self._params_to_flattened_tensor(high_params)
        
        for i, layer in enumerate(self.layers):
            if i==0 or i==1: continue
            
            if type(layer)==ReLUZ:
                # If ReLU is at a cross-boundary, params length increases of 1
                x, low_params, high_params = layer(x, low_params, high_params)
                
                low_params, high_params = _flip_low_high(low_params, high_params)
                
                low_params = _abs_tensor(low_params)
                high_params = _abs_tensor(high_params)
                
                assert (low_params.sign()*high_params.sign()>=0).all()
            else:
                x = layer(x)
                low_params = layer(low_params)
                low_params = low_params - layer.bias

                high_params = layer(high_params)
                high_params = high_params - layer.bias
                
                low_params, high_params = _flip_low_high(low_params, high_params)

                low_params = _abs_tensor(low_params)
                high_params = _abs_tensor(high_params)
                
                assert (low_params.sign()*high_params.sign()>=0).all()
        
        return x, low_params, high_params


class ReLUConvZ(nn.Module):
    
    def __init__(self, device, input_size):
        super(ReLUConvZ, self).__init__()
        # Initializing lambdas
        self.lambdas = torch.ones(input_size, requires_grad = True, device=device)
        self.initialized = False  # This flag is set to True after the first 
                                  # forward propagation, which allows to 
                                  # initialize lambda as DeepZ 
                                  
    def _computeBounds(self, x, low_params, high_params):
        
        low_params_sum = torch.sum(low_params, axis = 0)
        high_params_sum = torch.sum(high_params, axis = 0)
        low_params_sum.unsqueeze(0)
        high_params_sum.unsqueeze(0)
        low = x - low_params_sum
        high = x + high_params_sum
        return low, high
    
    def _zonotope(self, low, high, params):
        new_params = self.lambdas*params
        mask_to_initial = torch.cat(new_params.size(0)*[low>0])
        new_params[mask_to_initial] = params[mask_to_initial]
        mask_to_zero = torch.cat(new_params.size(0)*[high<=0])
        new_params[mask_to_zero] = 0
        
        # Add params if and only if any cross boundary exists
        if ((low<0)*(high>0)).any().item():
            # Create new line with new parameters. For now they are all in the 
            # same channel.
            new_fused_line = torch.zeros((1, params.size(1), params.size(2), params.size(3)))
            new_fused_line = (-0.5*self.lambdas*low)
            # Compute params in special case 0 <= lambda < high/(high-low)
            mask_lambdas = (self.lambdas) < (high/(high-low))
            new_fused_line[mask_lambdas] = new_fused_line[mask_lambdas] - 0.5*(-high+self.lambdas*(high-low))[mask_lambdas]
            # Concatenate as many new_fused_lines as many cross boundaries we have
            new_fused_line = torch.cat([new_fused_line for i in range(torch.sum((low<0)*(high>0)).item())])
            # Create a mask to put to preserve for each channel of new_fused_line only the corresponding new param
            zeros = torch.zeros_like(new_fused_line)
            indexes =  [tuple(torch.nonzero((low<0)*(high>0))[i].numpy()) for i in range(torch.sum((low<0)*(high>0)).item())]
            indexes =  [(idx[0]+i, idx[1], idx[2], idx[3]) for i, idx in enumerate(indexes)]
            for idx in indexes:
                zeros[idx] = 1
            # Set to zeros everything but the new param per channel
            new_fused_line = new_fused_line*zeros
            # Cat the so created new_fused_line with previously existing params
            new_params = torch.cat((new_params, new_fused_line))
        
        return new_params
    
    def forward(self, x, low_params, high_params):

        low, high = self._computeBounds(x, low_params, high_params) 
        assert (low<=high).all()  # Sanity check
        if not self.initialized:            
            self.lambdas.data = high/(high-low+1e-5)  # 1e-5 is used to avoid numerical issues (i.e. division by almost zero).
            self.initialized = True
        
        z = self.lambdas*(x-0.5*low)
        bias = -0.5*(-high+self.lambdas*(high-low))
        z[(self.lambdas<(high/(high-low)))] = (z + bias)[(self.lambdas<(high/(high-low)))]
        z[low>0] = x[low>0]
        z[high<=0] = 0
        
        new_low_params = self._zonotope(low, high, low_params)
        new_high_params = self._zonotope(low, high, high_params)
                
        return z, new_low_params, new_high_params 
    

class ConvZ(nn.Module):

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(ConvZ, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                ReLUConvZ(device, kernel_size),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [ReLUZ(device, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def _params_to_tensor(self, params):
        params_tensor = None
        for i, p in enumerate(params):  
            line = [0 for _ in params]
            line[i] = p
            line = torch.FloatTensor(line).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
            line = line/0.3081  # Normalize
                # Normalization on a delta means just dividing by the variance. 
                # Mean should not be substracted
            
            if params_tensor is None:
                params_tensor = line
            else:
                params_tensor = torch.cat((params_tensor, line), axis=0)
                    
        params = params_tensor
        return params

    def forward(self, x, low_params, high_params):
        
        x = self.layers[0](x)  # Normalize        
        
        low_params = self._params_to_tensor(low_params)
        high_params = self._params_to_tensor(high_params)
        
        for i, layer in enumerate(self.layers):
            if i==0: continue
            
            if type(layer)==ReLUConvZ or type(layer)==ReLUZ:
                # If ReLU is at a cross-boundary, params length increases of 1
                x, low_params, high_params = layer(x, low_params, high_params)
                
                low_params, high_params = _flip_low_high(low_params, high_params)
                
                low_params = _abs_tensor(low_params)
                high_params = _abs_tensor(high_params)
            elif type(layer) == nn.Flatten:
                x = layer(x)
                low_params = layer(low_params)
                high_params = layer(high_params)
            elif type(layer) == nn.Linear:
                x = layer(x)
                low_params = layer(low_params)
                high_params = layer(high_params)

                low_params = low_params - layer.bias
                high_params = high_params - layer.bias
                
                low_params, high_params = _flip_low_high(low_params, high_params)
                
                low_params = _abs_tensor(low_params)
                high_params = _abs_tensor(high_params)
            else:
                x = layer(x)
                
                bias = layer.bias
                bias = bias.unsqueeze(0)
                bias = bias.unsqueeze(2)
                bias = bias.unsqueeze(3)
                
                low_params = layer(low_params)
                high_params = layer(high_params)
                
                low_params = low_params - bias
                high_params = high_params - bias
                
                low_params, high_params = _flip_low_high(low_params, high_params)
                
                low_params = _abs_tensor(low_params)
                high_params = _abs_tensor(high_params)
        
        return x, low_params, high_params
