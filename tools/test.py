# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, _load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import ipdb
from tqdm import tqdm
from loguru import logger
import sys
from torch.nn import Linear, Conv2d
from _OBS import TrueOBS, Quantizer, get_module, _calibrate_input, _quantize_input, is_match
        
def logger_enable(prefix=''):
    def console_filter(record):
        # extra에 file_only가 True인 경우 콘솔 출력 제외
        return not record["extra"].get("file_only", False)
    global logger
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add(sys.stdout, level="INFO", format=LOG_FORMAT, filter=console_filter)
    logger.add("_logs/log", rotation="500 MB", level="INFO", format=LOG_FORMAT)
    logger = logger.bind(prefix=prefix)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--include", type=str)
    parser.add_argument("--exclude", type=str)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    distributed = None
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if distributed: 
            samples_per_gpu = 1
        else:
            samples_per_gpu = 4
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if True: # args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if distributed and args.prefix!='':      
        print(f'prefix:{args.prefix}')              
        params = _load_checkpoint(f'ckpts/{args.prefix}.pth', 'cpu', None)
        model.load_state_dict(params, strict=False)
        # w_scale, act_scale, max_q, min_q 추가
        for key_to_add, value_to_add in params.items():
            path_parts = key_to_add.split('.') 
            buffer_name_to_register = path_parts[-1]
            module_path_str = '.'.join(path_parts[:-1])
            if buffer_name_to_register not in ['w_scale','act_scale','maxq','minq']: continue
            module = get_module(model,module_path_str)
            module.register_buffer(buffer_name_to_register, value_to_add.clone().detach())

        for name, m in model.named_modules():
            if not isinstance(m,(Linear,Conv2d)): continue
            if not hasattr(m,'act_scale') : continue
            print(f'register prehook : {name}')
            m.register_forward_pre_hook(_quantize_input)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        logger_enable(args.prefix)
        BATCH_TO_PROCESS = 700
        BIT = 8
        damp = 0.01
        trueobs = {}

        model = MMDataParallel(model, device_ids=[0])
        model.eval() 
        
        exclude = ['re:.*pts_bbox_head.transformer.decoder.layers.*.attentions.0.attn.out_proj']
        include = []
        if args.include is not None:
            include.extend([ x for x in args.include.replace(' ','').split(',') ]) 
        if args.exclude is not None:
            exclude.extend([ x for x in args.exclude.replace(' ','').split(',') ]) 
        logger.info(f'INCLUDE: {include}')
        logger.info(f'EXCLUDE: {exclude}')
        PASS_INCLUDE= True if not include else False
        TARGET = []
        for name, m in model.named_modules():
            if not isinstance(m,(Linear,Conv2d)):continue
            if is_match(name,exclude): 
                print(name)
                continue
            if (not PASS_INCLUDE) and (not is_match(name, include)): continue
            print(name)
            TARGET.append(name)        
        logger.info(f'target modules: {TARGET}')
                    
        for name in TARGET:
            m = get_module(model,name)
            trueobs[name] = TrueOBS(m, rel_damp=damp)
            trueobs[name].quantizer = Quantizer()
            trueobs[name].quantizer.configure(
                bits=BIT, perchannel=True, sym=True, mse=False
            )


        def add_batch(name):
            def tmp(layer, inp, out):
                trueobs[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in TARGET:
            m = get_module(model,name)
            handles.append(m.register_forward_hook(add_batch(name)))
        logger.info('inference for Hessian start')
        for i, data in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
            logger.info(f'batch {i+1}/{BATCH_TO_PROCESS}')
            if i+1==BATCH_TO_PROCESS : break
        logger.info('inference for Hessian end')
        for h in handles:
            h.remove()    
        logger.info('Weight quantization start')
        DEAD_LAYER = []

        for i, name in enumerate(tqdm(TARGET,desc='weight Q')):
            m = get_module(model,name)
            try:
                error = trueobs[name].quantize()
                if error==0.0:
                    logger.info(f'{i+1:3}/{len(TARGET)} {name} : error 0 -> DEAD')
                    DEAD_LAYER.append(name)
                else:
                    logger.info(f'{i+1:3}/{len(TARGET)} {name} : {error}')
                    m = get_module(model,name)
                    m.register_buffer('w_scale',trueobs[name].quantizer.scale)
            except:
                DEAD_LAYER.append(name)
                logger.info(f'{i+1:3}/{len(TARGET)} {name} : DEAD')
            finally:
                trueobs[name].free()
        logger.info('Weight quantization end')

        # Act quantization
        handles = []
        for name in TARGET:
            m = get_module(model,name)
            if not isinstance(m,(Linear,Conv2d)): continue
            if name in DEAD_LAYER: continue
            m.register_buffer('maxq',torch.tensor(2**(BIT-1)-1, device=m.weight.device))
            m.register_buffer('minq',torch.tensor(-2**(BIT-1),  device=m.weight.device))
            handles.append(m.register_forward_pre_hook(_calibrate_input))   
        logger.info('Act quantization start')
        for i, data in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
            logger.info(f'batch {i+1}/{BATCH_TO_PROCESS}')
            if i+1==BATCH_TO_PROCESS : break
        for h in handles:
            h.remove() 
        logger.info('Act quantization end')        
        param = model.module.state_dict()
        torch.save(param, f'ckpts/{args.prefix}.pth')
        logger.info('Finished')    
        



    else:
        # ipdb.set_trace()
        # head = model.pts_bbox_head
        # for name, _ in head.named_children():
        #     print(name)    
        #     # loss_cls
        #     # loss_bbox
        #     # loss_iou
        #     # activate
        #     # positional_encoding
        #     # transformer
        #     # cls_branches
        #     # reg_branches
        #     # bev_embedding
        #     # query_embedding
            
        # for name, _ in model.named_children():
        #     print(name)    
        # # pts_bbox_head   projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead
        # # img_backbone    mmdet.models.backbones.resnet.ResNet
        # # img_neck        mmdet.models.necks.fpn.FPN
        # # grid_mask       projects.mmdet3d_plugin.models.utils.grid_mask.GridMask'
        # for name,_ in model.pts_bbox_head.transformer.named_children():print(name)
        # encoder
        # decoder
        # reference_points
        # can_bus_mlp


        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                assert False
                #mmcv.dump(outputs['bbox_results'], args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
                '/')[-1].split('.')[-2], args.prefix)
            if args.format_only:
                dataset.format_results(outputs, **kwargs)

            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))

                print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
