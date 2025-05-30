#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# PYTHONPATH=$(pwd):$PYTHONPATH python ./tools/test.py ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth --eval bbox --prefix test1 --include re:.*pts_bbox_head.transformer.decoder.layers.*.attentions.0.* --launcher none