"""
Implementation of U-Net architecture in Pytorch.
Architecture is constructed using blocks of layers.
Unlike the original implementation by Ronneberger et al., padded convolutions are used.
Batch normalization can be turned on or off, and is performed after every convolutional layer but before the non-linear activation function.
"""

import torch
import torch.nn as nn
from torchinfo import summary

class Block(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, batch_norm: bool = False) -> None:
        """
        Args:
            input_channels:  number of channels of the input tensor.
            output_channels:  number of channels of the output tensor.
            batch_norm:  specifies if batch normalization layers should be used.
        """
        super().__init__()
        # define state
        self.batch_norm = batch_norm

        # define layers
        # turn of bias for convolutional layers when batch normalization is used
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias= not self.batch_norm)
        if self.batch_norm == True:
            self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias= not self.batch_norm)
        if self.batch_norm == True:
            self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  input tensor
        Returns:
            x:  tensor after operations
        """
        x = self.conv1(x)
        x = self.bn1(x) if self.batch_norm == True else x
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batch_norm == True else x
        x = self.relu2(x)
        return x


class Down(Block):

    def __init__(self, input_channels: int, output_channels: int, downsample_method: str = 'max_pool', batch_norm: bool = False) -> None:
        """
        Args:
            input_channels:  number of channels of the input tensor.
            output_channels:  number of channels of the output tensor.
            downsample_method:  method to downsample feature maps.  
            batch_norm:  specifies if batch normalization layers should be used.
        """
        super().__init__(input_channels, output_channels, batch_norm)

        # define downsampling layer as strided convolution
        if downsample_method == 'max_pool':
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_method == 'strided_conv':
            self.downsample = nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2, padding=0)
        elif downsample_method == 'interpolate':
            self.downsample = lambda x: nn.functional.interpolate(x, scale_factor=0.5, mode='nearest')
        else:
            raise ValueError('Invalid argument for downsample method')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  input tensor
        Returns:
            x:  tensor after operations
        """
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x) if self.batch_norm == True else x
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batch_norm == True else x
        x = self.relu2(x)
        return x


class Up(Block):

    def __init__(self, input_channels: int, output_channels: int, upsample_method: str = 'interpolate', batch_norm: bool = False) -> None:
        """
        Args:
            input_channels:  number of channels of the input tensor.
            output_channels:  number of channels of the output tensor.
            upsample_method:  method to upsample feature maps.
            batch_norm:  specifies if batch normalization layers should be used.
        """
        # since upsampling does not reduce the number of feature maps, we need to update the expected number of input channels
        if upsample_method == 'interpolate':
            input_channels = int(1.5*input_channels)

        super().__init__(input_channels, output_channels, batch_norm)

        # define additional layers
        if upsample_method == 'transposed_conv':
            self.upsample = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=2, stride=2)
        elif upsample_method == 'interpolate':
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')    
        else:
            raise ValueError('Invalid argument for upsample method')       


    def forward(self, x_down: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_down:  input tensor from downsampling path though shortcut.
            x:  input tensor from upsampling path.
        Returns:
            x:  tensor after operations
        """
        x = self.upsample(x)
        x = torch.cat([x_down, x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x) if self.batch_norm == True else x
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batch_norm == True else x
        x = self.relu2(x)
        return x


class UNet(nn.Module):

    def __init__(
        self,
        input_channels: int,
        N_classes: int,
        filters: int = 8,
        downsample_method: str = 'max_pool',
        upsample_method: str = 'interpolate',
        batch_norm: bool = True,
        weight_init: str = 'kaiming_normal'
    ) -> None:
        """
        Args:
            input_channels:  number of channels of the input tensor.
            N_classes:  number of channels of the output tensor.
            filters:  number of filters used in the first convolutional layer. 
                Each consecutive layer in the encoder path uses twice as many filters.
                Each consecutive layer in the decoder path uses half as many filters.
            downsample_method:  method to downsample feature maps.  
            upsample_method:  method to upsample feature maps.
            batch_norm:  specifies if batch normalization should be applied.
            weight_init:  specifies which weight initialization method should be used.
        """
        super().__init__()

        # define hyperparameters as instance attributes
        self.filters = filters
        self.downsample_method = downsample_method
        self.upsample_method = upsample_method
        self.batch_norm = batch_norm
        self.weight_init = weight_init

        #define network layers
        self.layers = nn.ModuleDict({
            'block': Block(        input_channels, int(      self.filters),                         self.batch_norm),
            'down1': Down(int(      self.filters), int(  2 * self.filters), self.downsample_method, self.batch_norm),
            'down2': Down(int(  2 * self.filters), int(  4 * self.filters), self.downsample_method, self.batch_norm),
            'down3': Down(int(  4 * self.filters), int(  8 * self.filters), self.downsample_method, self.batch_norm),
            'down4': Down(int(  8 * self.filters), int( 16 * self.filters), self.downsample_method, self.batch_norm),
            'down5': Down(int( 16 * self.filters), int( 32 * self.filters), self.downsample_method, self.batch_norm),
            'up1'  : Up(  int( 32 * self.filters), int( 16 * self.filters), self.upsample_method,   self.batch_norm),
            'up2'  : Up(  int( 16 * self.filters), int(  8 * self.filters), self.upsample_method,   self.batch_norm),
            'up3'  : Up(  int(  8 * self.filters), int(  4 * self.filters), self.upsample_method,   self.batch_norm),
            'up4'  : Up(  int(  4 * self.filters), int(  2 * self.filters), self.upsample_method,   self.batch_norm),
            'up5'  : Up(  int(  2 * self.filters), int(      self.filters), self.upsample_method,   self.batch_norm),
            'final_conv': nn.Conv2d(self.filters, N_classes, kernel_size=1, padding=0, stride=1)
        })

        # recursively apply the initialize_weights method to all convolutional layers to initialize weights
        self.layers.apply(self.initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  input tensor
        
        Returns:
            logit:  tensor after operations
        """
        x1 = self.layers['block'](x)
        x2 = self.layers['down1'](x1)
        x3 = self.layers['down2'](x2)
        x4 = self.layers['down3'](x3)
        x5 = self.layers['down4'](x4)
        x  = self.layers['down5'](x5)
        x  = self.layers['up1'](x5, x)
        x  = self.layers['up2'](x4, x)
        x  = self.layers['up3'](x3, x)
        x  = self.layers['up4'](x2, x)
        x  = self.layers['up5'](x1, x)
        logit = self.layers['final_conv'](x)
        return logit


    def initialize_weights(self, layer: torch.nn) -> None:
        """
        Initialize the weights using the specified initialization method
        if it is a 2D convolutional layer.
        Args:
            layer:  torch network layer
        """
        # define dictionary with initialization function and names
        init_methods = {'xavier_uniform' : nn.init.xavier_uniform_,
                        'xavier_normal'  : nn.init.xavier_normal_,
                        'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x, mode='fan_out', nonlinearity='relu'),
                        'kaiming_normal' : lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu'),
                        'zeros'          : nn.init.zeros_}

        if isinstance(layer, nn.Conv2d) == True:
            # select the correct weight initialization function and initialize the layer weights
            # subsequently initialize the biases to zero
            if self.weight_init in init_methods.keys():
                init_methods[self.weight_init](layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError('Initialization method not recognized.')


if __name__ == '__main__':
    # define U-Net network instance
    net = UNet(3, 2).to('cuda')
    input_shape = (1, 3, 256, 384)
    summary(net, input_shape, depth=3, col_names=["input_size", "output_size", "num_params"])