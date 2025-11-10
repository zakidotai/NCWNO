"""
NCWNO for Darcy in triangular domain with notch (2D problem)

It requires the package "Pytorch Wavelets"
-- see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html

"""

import numpy as np
import torch
import torch.nn as nn

from pytorch_wavelets import DWT1D, IDWT1D
from pytorch_wavelets import DWT, IDWT  
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import uqpy.scientific_machine_learning as sml

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 1d Wavelet convolution layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet 
        self.mode = mode
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode) 
        dummy_data = torch.randn( 1, 1, self.size ) 
        mode_data, _ = self.dwt_(dummy_data)
        self.modes1 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass wavelet coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)
    
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        x = idwt((out_ft, out_coeff)) 
        return x

""" Def: 1d Wavelet convolutional encoder layer """
class WaveEncoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, down_level=1, mode='symmetric'):
        super(WaveEncoder1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet
        self.mode = mode
        if down_level >= level:
            raise Exception('down_level must be smaller than level')
        else:
            self.down_level = down_level
        dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode) 
        dummy_data = torch.randn(1, 1, self.size )
        mode_data, _ = dwt_(dummy_data)
        self.modes = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes))

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass and high pass coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        out_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            x = idwt((out_ft, out_coeff[factor + self.down_level:])) 
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            x = idwt((out_ft, out_coeff[factor - self.down_level:])) 
            
        else:
            x = idwt((out_ft, out_coeff[self.down_level:]))                
        return x
    

""" Def: Gate Network """
class Gate_context1d(nn.Module):
    def __init__(self, in_channels, out_channels, expert_num, label_lifting, size, level=2, wavelet='db1', down_level=1, is_bayesian=False):
        super(Gate_context1d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        expert_num   : scalar, number of local wavelet experts 
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        
        Output parameters:
        ------------------
        lambda : tensor, shape-[in_channels * out_channels * number of expert]
                  participation coefficients of local experts
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.level = level
        self.size = size 
        self.wavelet = wavelet
        self.down_level = down_level
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.lifting_network = nn.Linear(1, self.label_lifting)
        self.wno_encode = WaveEncoder1d(self.in_channels, self.out_channels, self.level,
                                        self.size, self.wavelet, self.down_level)

        if is_bayesian:
            self.gate = nn.Sequential(
                    sml.BayesianLinear(self.size//2**(down_level) + self.label_lifting, 256),
                    nn.Mish(),
                    sml.BayesianLinear(256, 128),
                    nn.Mish(),
                    sml.BayesianLinear(128, 64),
                    nn.Mish(),
                    sml.BayesianLinear(64, 32),
                    nn.Mish(),
                    sml.BayesianLinear(32, self.expert_num),
                    nn.Mish(),
                    nn.Softmax(dim=-1))
        else:
            self.gate = nn.Sequential(
                    nn.Linear(self.size//2**(down_level) + self.label_lifting, 256),
                    nn.Mish(),
                    nn.Linear(256, 128),
                    nn.Mish(),
                    nn.Linear(128, 64),
                    nn.Mish(),
                    nn.Linear(64, 32),
                    nn.Mish(),
                    nn.Linear(32, self.expert_num),
                    nn.Softmax(dim=-1))
        
    def forward(self, x, label):
        lambda_0 = self.lifting_network(label)
        lambda_1 = self.wno_encode(x)

        lambda_ = self.gate( torch.cat((lambda_0,lambda_1), dim=-1) )
        return lambda_     

    
""" Def: 2d Wavelet convolution layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WaveConv2d, self).__init__()
        
        """
        2D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : list, domain of input 2D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2, self.weights3, self.weights4, 
                      : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet) 
        dummy_data = torch.randn( 1,1,*self.size ) 
        mode_data, _ = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, *x_ft.shape[2:], device= x.device)
        out_coeff = [torch.zeros(coeffs.shape[0], self.out_channels, *coeffs.shape[2:], device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

""" Def: 2d Wavelet convolutional Encoder layer """
class WaveEncoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, down_level=1, mode='symmetric'):
        super(WaveEncoder2d, self).__init__()

        """
        2D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : list, domain of input 2D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        if down_level >= level:
            raise Exception('down_level must be smaller than level')
        else:
            self.down_level = down_level
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet) 
        dummy_data = torch.randn( 1,1,*self.size ) 
        mode_data, _ = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, *x_ft.shape[2:], device= x.device)
        out_coeff = [torch.zeros(coeffs.shape[0], self.out_channels, *coeffs.shape[2:], device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            x = idwt((out_ft, out_coeff[factor + self.down_level:])) 
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            x = idwt((out_ft, out_coeff[factor - self.down_level:])) 
            
        else:
            x = idwt((out_ft, out_coeff[self.down_level:]))                
        return x
    
""" Def: Gate Network """
class Gate_context2d(nn.Module):
    def __init__(self, in_channels, out_channels, expert_num, label_lifting, size, level=2, wavelet='db1', down_level=1):
        super(Gate_context2d, self).__init__()

        """
        Probabilistic Gate for expert mixing 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        expert_num   : scalar, number of local wavelet experts 
        size         : list, domain of input 2D signal 
        wavelet      : string, wavelet filters
        
        Output parameters:
        ------------------
        lambda : tensor, shape-[in_channels * out_channels * number of expert]
                 participation coefficients of local experts
        """

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.level = level
        self.size = size 
        self.wavelet = wavelet
        self.down_level = down_level
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.lifting_network = nn.Linear(1, self.label_lifting)
        self.flatten_size = self.size[0]//2**(down_level) * self.size[1]//2**(down_level)
        self.wno_encode = nn.Sequential(
                                        WaveEncoder2d(self.in_channels, self.out_channels, 
                                        self.level, self.size, self.wavelet, self.down_level),
                                        nn.Mish(),
                                        WaveEncoder2d(self.out_channels, self.out_channels, self.level,
                                                      [self.size[0]//2**(down_level), self.size[1]//2**(down_level)],
                                                      self.wavelet, self.down_level) )
        
        self.gate = nn.Sequential(nn.Linear(self.flatten_size + self.label_lifting, 256),
                                  nn.Mish(),
                                  nn.Linear(256, 128),
                                  nn.Mish(),
                                  nn.Linear(128, 64),
                                  nn.Mish(),
                                  nn.Linear(64, 32),
                                  nn.Mish(),
                                  nn.Linear(32, self.expert_num),
                                  nn.Softmax(dim=-1))
        
    def forward(self, x, label):
        lambda_0 = self.lifting_network(label)
        lambda_1 = self.wno_encode(x).flatten(2,3) 
        
        lambda_ = self.gate( torch.cat((lambda_0,lambda_1), dim=-1) )
        return lambda_.unsqueeze(-1)   


""" Def: 2d Wavelet convolution layer (Slim Continuous Wavelet decomposition) """
class WaveConv2dcwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1='near_sym_a', wavelet2='qshift_a'):
        super(WaveConv2dcwt, self).__init__()
        
        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : list, domain of input 2D signal 
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2  
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2) 
        dummy = torch.randn(1, 1, *size)        
        mode_data, mode_coef = dwt_(dummy)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.mul2d(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights15r)
        out_coeff[-1][:,:,0,:,:,1] = self.mul2d(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights15c)
        out_coeff[-1][:,:,1,:,:,0] = self.mul2d(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights45r)
        out_coeff[-1][:,:,1,:,:,1] = self.mul2d(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights45c)
        out_coeff[-1][:,:,2,:,:,0] = self.mul2d(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights75r)
        out_coeff[-1][:,:,2,:,:,1] = self.mul2d(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights75c)
        out_coeff[-1][:,:,3,:,:,0] = self.mul2d(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights105r)
        out_coeff[-1][:,:,3,:,:,1] = self.mul2d(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights105c)
        out_coeff[-1][:,:,4,:,:,0] = self.mul2d(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights135r)
        out_coeff[-1][:,:,4,:,:,1] = self.mul2d(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights135c)
        out_coeff[-1][:,:,5,:,:,0] = self.mul2d(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights165r)
        out_coeff[-1][:,:,5,:,:,1] = self.mul2d(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights165c)
        
        # Return to physical space        
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x

""" Def: 2d Wavelet convolutional Encoder layer """
class WaveEncoder2dcwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, down_level=1, mode='symmetric'):
        super(WaveEncoder2dcwt, self).__init__()

        """
        2D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : list, domain of input 2D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        if down_level >= level:
            raise Exception('down_level must be smaller than level')
        else:
            self.down_level = down_level
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet) 
        dummy_data = torch.randn( 1,1,*self.size ) 
        mode_data, _ = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, *x_ft.shape[2:], device= x.device)
        out_coeff = [torch.zeros(coeffs.shape[0], self.out_channels, *coeffs.shape[2:], device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            x = idwt((out_ft, out_coeff[factor + self.down_level:])) 
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            x = idwt((out_ft, out_coeff[factor - self.down_level:])) 
            
        else:
            x = idwt((out_ft, out_coeff[self.down_level:]))                
        return x
    

""" Def: Gate Network """
class Gate_context2dcwt(nn.Module):
    def __init__(self, in_channels, out_channels, expert_num, label_lifting, size, level=2, wavelet='db1', down_level=1):
        super(Gate_context2dcwt, self).__init__()

        """
        Probabilistic Gate for expert mixing 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        expert_num   : scalar, number of local wavelet experts 
        size         : list, domain of input 2D signal 
        wavelet      : string, wavelet filters
        
        Output parameters:
        ------------------
        lambda : tensor, shape-[in_channels * out_channels * number of expert]
                 participation coefficients of local experts
        """

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.level = level
        self.size = size 
        self.wavelet = wavelet
        self.down_level = down_level
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.lifting_network = nn.Linear(1, self.label_lifting)
        self.flatten_size = self.size[0]//2**(down_level + 1) * self.size[1]//2**(down_level + 1)
        self.wno_encode = nn.Sequential(
                                        WaveEncoder2d(self.in_channels, self.out_channels, 
                                        self.level, self.size, self.wavelet, self.down_level),
                                        nn.Mish(),
                                        WaveEncoder2d(self.out_channels, self.out_channels, self.level,
                                                      [self.size[0]//2**(down_level), self.size[1]//2**(down_level)],
                                                      self.wavelet, self.down_level) )
        
        self.gate = nn.Sequential(nn.Linear(self.flatten_size + self.label_lifting, 256),
                                  nn.Mish(),
                                  nn.Linear(256, 128),
                                  nn.Mish(),
                                  nn.Linear(128, 64),
                                  nn.Mish(),
                                  nn.Linear(64, 32),
                                  nn.Mish(),
                                  nn.Linear(32, self.expert_num),
                                  nn.Softmax(dim=-1))
        
    def forward(self, x, label):
        lambda_0 = self.lifting_network(label)
        lambda_1 = self.wno_encode(x).flatten(2,3) 
        
        lambda_ = self.gate( torch.cat((lambda_0,lambda_1), dim=-1) )
        return lambda_.unsqueeze(-1)       
