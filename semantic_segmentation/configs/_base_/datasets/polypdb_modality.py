# dataset settings
dataset_type = 'PolypDBDataset'
data_root = "/mnt/b33c377d-a988-494e-860f-8149fffe7254/yangguojing/test/data/PolypDB/modality_wise"
# data_root = "/mnt/b33c377d-a988-494e-860f-8149fffe7254/yangguojing/vis"

img_scale = (512,512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path="train/images",
                seg_map_path="train/masks"),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="val/images",
            seg_map_path="val/masks"),
        pipeline=test_pipeline))

test_dataloader = dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path="test/WLI/images",
                seg_map_path="test/WLI/masks"),
            pipeline=test_pipeline))


# mmsegmentation/mmseg/evaluation/metrics/iou_metric.py
val_evaluator = dict(type='IoUMetric', iou_metrics=['polyp_metrics'])
test_evaluator = val_evaluator
