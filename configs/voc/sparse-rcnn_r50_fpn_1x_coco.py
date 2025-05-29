METAINFO = dict(
    classes=(
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ),
    palette=[
        (
            106,
            0,
            228,
        ),
        (
            119,
            11,
            32,
        ),
        (
            165,
            42,
            42,
        ),
        (
            0,
            0,
            192,
        ),
        (
            197,
            226,
            255,
        ),
        (
            0,
            60,
            100,
        ),
        (
            0,
            0,
            142,
        ),
        (
            255,
            77,
            255,
        ),
        (
            153,
            69,
            1,
        ),
        (
            120,
            166,
            157,
        ),
        (
            0,
            182,
            199,
        ),
        (
            0,
            226,
            252,
        ),
        (
            182,
            182,
            255,
        ),
        (
            0,
            0,
            230,
        ),
        (
            220,
            20,
            60,
        ),
        (
            163,
            255,
            0,
        ),
        (
            0,
            82,
            0,
        ),
        (
            3,
            95,
            161,
        ),
        (
            0,
            80,
            100,
        ),
        (
            183,
            130,
            88,
        ),
    ])
auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
data_root = 'data/VOCdevkit/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True, show=True, type='DetVisualizationHook', wait_time=2))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/sparse-rcnn_r50_fpn_1x_voc_10/epoch_57.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=20,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=6,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        type='SparseRoIHead'),
    rpn_head=dict(
        num_proposals=100,
        proposal_feature_channel=256,
        type='EmbeddingRPNHead'),
    test_cfg=dict(rcnn=dict(max_per_img=100), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
        ],
        rpn=None),
    type='SparseRCNN')
num_proposals = 100
num_stages = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=0.0001, momentum=0.9, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=70,
        gamma=0.1,
        milestones=[
            50,
            60,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=91)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='voc07_test.json',
        data_prefix=dict(img=''),
        data_root='data/VOCdevkit/',
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            ),
            palette=[
                (
                    106,
                    0,
                    228,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    197,
                    226,
                    255,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    153,
                    69,
                    1,
                ),
                (
                    120,
                    166,
                    157,
                ),
                (
                    0,
                    182,
                    199,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    163,
                    255,
                    0,
                ),
                (
                    0,
                    82,
                    0,
                ),
                (
                    3,
                    95,
                    161,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    183,
                    130,
                    88,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/VOCdevkit/voc07_test.json',
    format_only=True,
    metric=[
        'bbox',
        'segm',
    ],
    outfile_prefix='./work_dirs/voc07/test',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1000,
        600,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=48, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file='voc07_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/VOCdevkit/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            ),
            palette=[
                (
                    106,
                    0,
                    228,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    197,
                    226,
                    255,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    153,
                    69,
                    1,
                ),
                (
                    120,
                    166,
                    157,
                ),
                (
                    0,
                    182,
                    199,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    163,
                    255,
                    0,
                ),
                (
                    0,
                    82,
                    0,
                ),
                (
                    3,
                    95,
                    161,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    183,
                    130,
                    88,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1000,
        600,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='voc07_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/VOCdevkit/',
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            ),
            palette=[
                (
                    106,
                    0,
                    228,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    197,
                    226,
                    255,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    153,
                    69,
                    1,
                ),
                (
                    120,
                    166,
                    157,
                ),
                (
                    0,
                    182,
                    199,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    163,
                    255,
                    0,
                ),
                (
                    0,
                    82,
                    0,
                ),
                (
                    3,
                    95,
                    161,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    183,
                    130,
                    88,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/VOCdevkit/voc07_val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs/sparse-rcnn_r50_fpn_1x_voc_10'
