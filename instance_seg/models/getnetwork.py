"""
Network factory function.

Simplified version from Semi-MoE that only includes networks we use.
"""

import sys
from .networks.unet import unet, r2_unet, attention_unet


def get_network(network, in_channels, num_classes, **kwargs):
    """
    Get network by name.
    
    Parameters
    ----------
    network : str
        Network name (e.g., 'unet', 'r2unet', 'attunet')
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes/channels
    **kwargs : dict
        Additional network-specific arguments
        
    Returns
    -------
    torch.nn.Module
        Network model
    """
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'r2unet':
        net = r2_unet(in_channels, num_classes)
    elif network == 'attunet':
        net = attention_unet(in_channels, num_classes)
    else:
        print(f'Network "{network}" is not supported. Available: unet, r2unet, attunet')
        sys.exit()
    
    return net
