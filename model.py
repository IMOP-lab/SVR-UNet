from torch import nn
from monai.networks.nets import BasicUNet
from monai.networks.nets import UNETR
from networks.UXNet_3D.network_backbone import UXNET
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.SwinUNETR.SwinUNETR import SwinUNETR
from networks.mednext.MedNext import MedNeXt
from monai.networks.nets.swin_unetr import SwinUNETR
from networks.UNesT.unest import UNesT
from networks.SwinSMT.src.models.swin_smt import SwinSMT
from networks.nnWNet.nnWNet import WNet3D
from networks.SuperLightNet.superlightnet import NormalU_Net
from networks.VSmTrans.VSmTrans import VSmixTUnet
from networks.SegMamba.segmamba import SegMamba
from networks.SVRUNet.SVRUNet import SVRUNet


def get3dmodel(network, in_channel, out_classes):
    ## UNet
    if network == 'UNet':
        model = BasicUNet(in_channels=in_channel, out_channels=out_classes)
        
    elif network=='SVRUNet':
        model = SVRUNet(
             in_channels=in_channel,
             out_channels=out_classes,
        )
    
        
    ## UNETR
    elif network == 'UNETR':
        model = UNETR(
            in_channels=in_channel,
            out_channels=out_classes,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0)
        
        
    ## 3DUXNET
    elif network == '3DUXNET':
        model = UXNET(
            in_chans=in_channel,
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3)
  
    ## nnFormer
    elif network == 'nnFormer':
        model = nnFormer(
            input_channels=in_channel, 
            num_classes=out_classes)      
        
        
    ## SwinUNETR 
    elif network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False)


    elif network == 'UNesT':
        model = UNesT(
            in_channels=in_channel,
            out_channels=out_classes
        )
    
    elif network == 'SwinSMT':
        model = SwinSMT(
            img_size=(96,96,96),
            in_channels=in_channel,
            out_channels=out_classes
        )
        
   
    elif network == 'MedNeXt':
        model = MedNeXt(
            in_channels=in_channel,
            n_channels=32,
            n_classes=out_classes
        )

    elif network == 'nnWNet':
        model = WNet3D(
            in_channel=in_channel,
            num_classes=out_classes,
        )

    elif network == 'SuperLightNet':
        model = NormalU_Net(
            init_channels=in_channel,
            class_nums=out_classes,
            depths_unidirectional='small',
        )

    elif network =='VSmTrans':
        model = VSmixTUnet(
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            split_size=[1, 3, 5, 7],
            window_size=7,
            num_heads=[3, 6, 12, 24],
            img_size=[96, 96, 96],
            depths=[2, 2, 2, 2],
            patch_size=(2, 2, 2),
        )
        
    
    elif network == 'SegMamba':
       model = SegMamba(
           in_chans=in_channel,
           out_chans=out_classes
       )

        
    return model
