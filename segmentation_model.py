import segmentation_models_pytorch as smp

def get_unet(in_channels:int, out_channels:int, encoder_name='resnet34', activation='sigmoid'):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=out_channels,
        activation=activation
    )