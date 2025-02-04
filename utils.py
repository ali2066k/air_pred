import math
import torch
from monai.networks.nets import UNet, AttentionUnet

def get_model(params, device="cpu"):
    """
    Initialize and return a model based on given parameters.

    Args:
        params (dict): Model configuration parameters.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The initialized model.
    """
    co = int(math.log(params['CHANNELS'], 2))
    channels = tuple((2 ** j) for j in range(co, co + params['N_LAYERS']))
    strides = ((params['STRIDES']),) * params['N_LAYERS']

    match params['MODEL_NAME']:
        case 'ResUNet':
            model = UNet(
                spatial_dims=params['SPATIAL_DIMS'],
                in_channels=params['IN_CHANNELS'],
                out_channels=params['OUT_CHANNELS'],
                channels=channels,
                strides=strides,
                num_res_units=params['NUM_RES_UNITS'],
                dropout=params['DROPOUT'],
            )
        case 'AttentionUNet':
            model = AttentionUnet(
                spatial_dims=params['SPATIAL_DIMS'],
                in_channels=params['IN_CHANNELS'],
                out_channels=params['OUT_CHANNELS'],
                channels=channels,
                strides=strides,
                kernel_size=[3, 3, 3],
                up_kernel_size=[3, 3, 3],
                dropout=params['DROPOUT'],
            )
        case 'BasicUNet':
            model = UNet(
                spatial_dims=params['SPATIAL_DIMS'],
                in_channels=params['IN_CHANNELS'],
                out_channels=params['OUT_CHANNELS'],
                channels=channels,
                strides=strides,
                num_res_units=params.get('NUM_RES_UNITS', 2),
                dropout=params['DROPOUT'],
            )
        case _:
            raise ValueError(f"Unsupported model name: {params['MODEL_NAME']}")

    return model
    # return model.to(device)