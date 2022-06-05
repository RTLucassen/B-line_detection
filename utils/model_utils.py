"""
Implementation of utility functions for models.
"""

import torch
import torch.nn as nn


def change_input(layer: nn.modules.conv, input_channels: int) -> nn.modules.conv:
    """
    Args:
        layer:  convolutional network layer.
        input_channels:  number of input channels that the returned layer should have.
    
    Returns:
        modified_layer:  convolutional layer with same characteristics except for the changed number of input channels.
    """
    # check the type of convolutional layer
    if isinstance(layer, nn.modules.conv.Conv2d):
        Conv = nn.Conv2d
    elif isinstance(layer, nn.modules.conv.Conv3d):
        Conv = nn.Conv3d
    else:
        raise ValueError(f'Convolutional layer type {type(layer)} was unrecognized.')

    # create an equivalent layer with only the input channels changed
    modified_layer = Conv(
        in_channels=input_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=False if layer.bias == None else True
        )
    return modified_layer

def initialize(layer: torch.nn, method: str = 'kaiming_normal') -> None:
        """
        Initialize the weights and biases using the specified initialization method.

        Args:
            layer:  torch network layer
            method:  initialization method
        """
        # create a dictionary with initialization function and names
        init_methods = {
            'xavier_uniform' : nn.init.xavier_uniform_,
            'xavier_normal'  : nn.init.xavier_normal_,
            'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x, mode='fan_in', nonlinearity='relu'),
            'kaiming_normal' : lambda x: nn.init.kaiming_normal_(x, mode='fan_in', nonlinearity='relu'),
        }
        # select the correct initialization method
        if method in init_methods.keys():
            init_method = init_methods[method]
        else:
            raise ValueError('Initialization method not recognized.')

        # perform the initialization based on the layer type
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            init_method(layer.weight)
            if layer.bias != None:
                nn.init.constant_(layer.bias, 0)
                
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)