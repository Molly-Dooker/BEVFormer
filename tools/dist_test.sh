#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$(pwd):$PYTHONPATH python ./tools/test.py ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth --eval bbox --prefix 'test0' --include 're:.*pts_bbox_head.transformer.encoder.layers.5.attentions, re:.*img_backbone.layer1.*' --launcher none