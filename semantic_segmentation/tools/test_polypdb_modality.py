# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import copy
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('--config', default='semantic_segmentation/configs/tsbanet/mamba_vision_tsba_head_20k_polypdb_modality_512x512_base.py', help='train config file path')
    parser.add_argument('--checkpoint', default='polypdb/iter_20000.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default='paper/test_result/polypdb',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # multi_test_datasets
    datasets = {
        "BLI": dict(
            data_prefix=dict(
                img_path="test/BLI/images",
                seg_map_path="test/BLI/masks"
            )
        ),
        "WLI": dict(
            data_prefix=dict(
                img_path="test/WLI/images",
                seg_map_path="test/WLI/masks"
            )
        ),
        "FICE": dict(
            data_prefix=dict(
                img_path="test/FICE/images",
                seg_map_path="test/FICE/masks"
            )
        ),
        "LCI": dict(
            data_prefix=dict(
                img_path="test/LCI/images",
                seg_map_path="test/LCI/masks"
            )
        ),
        "NBI": dict(
            data_prefix=dict(
                img_path="test/NBI/images",
                seg_map_path="test/NBI/masks"
            )
        )
    }

    for name, dataset_cfg in datasets.items():

        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.test_dataloader.dataset.data_prefix = dataset_cfg["data_prefix"]

        # dataset work_dir
        cfg_tmp.work_dir = osp.join(cfg.work_dir, name)

        if args.out:
            cfg_tmp.test_evaluator['output_dir'] = osp.join(args.out, name)

        # build the runner from config
        runner = Runner.from_cfg(cfg_tmp)

        # start testing
        runner.test()
    


if __name__ == '__main__':
    main()
