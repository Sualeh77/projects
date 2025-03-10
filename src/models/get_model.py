import segmentation_models_pytorch as smp
import torch

# Import utilities for MPS compatibility
from src.utils.utils import replace_segformer_upsampling
from src.config.base_config import patch_upsampling_mode, UPSAMPLING_MODE

def get_model(architecture, encoder, encoder_weights=None, encoder_depth=5,
              decoder_channels=(256, 128, 64, 32, 16), decoder_use_batchnorm=False,
              decoder_attention_type=None, in_channels=3, classes=1, activation=None,
              decoder_atrous_rates=None, decoder_aspp_separable=None, decoder_aspp_dropout=None, aux_params=None):
    """
    Get a segmentation model based on configuration.
    
    Args:
        architecture (str): Model architecture ('Unet', 'UnetPlusPlus', etc.)
        encoder (str): Encoder backbone ('resnet18', 'resnet34', etc.)
        encoder_weights (str): Pre-trained weights ('imagenet' or None)
        encoder_depth (int): Depth of the encoder
        decoder_channels (tuple): Number of channels in decoder blocks
        decoder_use_batchnorm (bool): Whether to use batch normalization in decoder
        decoder_attention_type (str): Type of attention to use in decoder
        in_channels (int): Number of input channels
        activation (str): Activation function for the output
    
    Returns:
        model: Initialized PyTorch model
    """
    
    if architecture.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture.lower() == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture.lower() == 'segformer':
        # Correct capitalization for Segformer (lowercase 's')
        model = smp.Segformer(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params
        )
        
        # Apply MPS compatibility fixes for SegFormer if needed
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS device detected. Replacing bicubic upsampling with {UPSAMPLING_MODE}...")
            model = replace_segformer_upsampling(model)
            model = patch_upsampling_mode(model)
            
    elif architecture.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            decoder_aspp_separable=decoder_aspp_separable,
            decoder_aspp_dropout=decoder_aspp_dropout,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model

def count_parameters(model):
    """
    Count the number of parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Dictionary containing parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for name, p in model.named_parameters() if 'encoder' in name)
    decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'decoder' in name)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'encoder': encoder_params,
        'decoder': decoder_params
    } 