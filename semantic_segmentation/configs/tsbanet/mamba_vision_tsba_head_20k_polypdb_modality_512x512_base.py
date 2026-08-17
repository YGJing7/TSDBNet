_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/polypdb_modality.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# Crop size for training and inference
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# Model configuration
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained="/home/gjyang/project/code/MambaMedSeg_our/ckpts/mambavision_base_1k.pth.tar",  # Pretrained weights
        depths=(3, 3, 10, 5),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 64, 32),
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.4,
        norm_layer="ln2d",
        layer_scale=1e-5,
    ),
    decode_head=dict(
        _delete_=True,  # Delete the default decode head config
        type='TSBAHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=2,
        mixer_cfg=dict(d_state=8, d_conv=3, expand=2),
        
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # dict(type='TverskyLoss', alpha=0.7, beta=0.3, loss_weight=1.0)
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4),
        align_corners=False
    )
)