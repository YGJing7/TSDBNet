# Optimizer configuration
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Use mixed-precision optimizer
    optimizer=dict(
        type='AdamW',  # AdamW optimizer
        lr=0.00001,  # Learning rate
        betas=(0.9, 0.999),  # Beta values for AdamW
        weight_decay=0.01  # Weight decay for regularization
    ),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)  # No weight decay for normalization layers
        }
    )
)

# Learning rate scheduler configuration
param_scheduler = [
    dict(
        type='LinearLR',  # Linear learning rate warmup
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',  # Polynomial decay learning rate
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False
    )
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')) #draw=True, interval=1
