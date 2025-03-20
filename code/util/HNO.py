import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)  # (seq_length, num_channels)
        return x
    
    
class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1=None, modes2=None):
        super(SpectralConv2d_Uno, self).__init__()
        """
        Adapted from https://github.com/ashiq24/UNO/blob/main/integral_operators.py

        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y (or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2 
        self.scale = (1/(2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale*(torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float)))
        self.weights2 = nn.Parameter(self.scale*(torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        weights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2//2+1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm='forward')
        return x

class pointwise_op_2D(nn.Module):
    """ 
    dim1 = Default output grid size along x (or 1st dimension) 
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """
    def __init__(self, in_codim, out_codim, dim1, dim2):
        super(pointwise_op_2D,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2

        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size=(dim1, dim2), mode='bicubic', align_corners=True, antialias=True)
        return x_out

class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1, modes2):
        super(OperatorBlock_2D, self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2)
        self.mlp = MLP(out_codim, out_codim, out_codim)
        #self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)
        self.normalize_layer = torch.nn.LayerNorm([dim1, dim2])

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        x1_out = self.conv(x, dim1, dim2)
        x1_out = self.normalize_layer(x1_out)
        x2_out = self.w(x, dim1, dim2)
        x_out = x1_out + x2_out
        x_out = F.gelu(x_out)
        x_out = self.mlp(x_out)
        x_out = self.normalize_layer(x_out)
        x_out = x_out + x2_out
        x_out = F.gelu(x_out)

        return x_out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
def kernel(in_chan=2, out_chan=4, up_dim=16):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, out_chan, bias=False)
            )
    return layers
    
class UNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, width, pad=0, factor=1, pad_both=False, d_pe=None):  # d_pe: dimension for positional encoding for each coordinate
        super(UNO2D, self).__init__()

        self.in_width = in_channels # input channel
        self.width = width 
        self.d_pe = d_pe
        self.padding = pad  # pad the domain if input is non-periodic
        self.pad_both = pad_both
        if self.d_pe is None:
            self.fc_n1 = nn.Linear(self.in_width+2, self.width//2)
        else:
            self.fc_n1 = nn.Linear(self.in_width+self.d_pe*2, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv1 = OperatorBlock_2D(self.width, 2*factor*self.width, 192, 96, 96, 48)
        self.conv2 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 128, 64, 64, 32)
        self.conv3 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 64, 32, 32, 16)
        self.conv4 = OperatorBlock_2D(8*factor*self.width, 16*factor*self.width, 32, 16, 16, 8)
        self.conv5 = OperatorBlock_2D(16*factor*self.width, 8*factor*self.width, 64, 32, 16, 8)
        self.conv6 = OperatorBlock_2D(16*factor*self.width, 4*factor*self.width, 128, 64, 32, 16) 
        self.conv7 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 192, 96, 64, 32)
        self.conv8 = OperatorBlock_2D(4*factor*self.width, 2*self.width, 256, 128, 96, 48)

        if self.d_pe is None:
            self.knet = kernel(3*self.width+2, 3*self.width, 3*self.width)
        else:
            self.knet = kernel(3*self.width+self.d_pe*2, 3*self.width, 3*self.width)

        self.fc1 = nn.Linear(6*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device, self.d_pe)
        x = torch.cat((x, grid), dim=-1)
        x_fc = self.fc_n1(x)
        x_fc = F.gelu(x_fc)
        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        x_c1 = self.conv1(x_fc0)
        x_c2 = self.conv2(x_c1)
        x_c3 = self.conv3(x_c2)
        x_c4 = self.conv4(x_c3)
        x_c5 = self.conv5(x_c4)
        x_c5 = torch.cat([x_c5, x_c3], dim=1)
        x_c6 = self.conv6(x_c5)
        x_c6 = torch.cat([x_c6, x_c2], dim=1)
        x_c7 = self.conv7(x_c6)
        x_c7 = torch.cat([x_c7, x_c1], dim=1)
        x_c8 = self.conv8(x_c7)
        x_c8 = torch.cat([x_c8, x_fc0], dim=1)

        x_c8 = x_c8.permute(0, 2, 3, 1)

        # L
        res2 = x_c8.shape[2]
        grid = self.get_grid(x_c8.shape, x_c8.device, self.d_pe)
        kx = self.knet(torch.cat((x_c8, grid), dim=-1))
        x_out = torch.einsum('bxzi,bxzi->bxi', kx, x_c8) / res2
        x_out = x_out.unsqueeze(2)
        
        # Q
        x_out = self.fc1(torch.cat((x_out, x_c8[:, :, [0], :]), dim=-1))
        x_out = F.gelu(x_out)
        x_out = self.fc2(x_out)
        
        return x_out
    
    def get_grid(self, shape, device, d_pe=None):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        if d_pe is None:
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(device)
        else:
            pe = PositionalEmbedding(num_channels=self.d_pe)
            gridx = torch.linspace(0, 2*torch.pi, size_x, dtype=torch.float)
            gridy = torch.linspace(0, 2*torch.pi, size_y, dtype=torch.float)
            gridx = pe(gridx)  # (size_x, num_channels)
            gridy = pe(gridy)  # (size_y, num_channels)
            gridx = gridx.reshape(1, size_x, 1, -1).repeat([batchsize, 1, size_y, 1])
            gridy = gridy.reshape(1, 1, size_y, -1).repeat([batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(device)
        
        return grid