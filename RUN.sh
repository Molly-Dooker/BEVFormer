#!/bin/bash

# --- 스크립트 설정 ---
CONFIG_FILE="./projects/configs/bevformer/bevformer_base.py"
CHECKPOINT_FILE="./ckpts/bevformer_r101_dcn_24ep.pth"
GPUS=8 # dist_test.sh에서 사용할 GPU 수

# INCLUDE 목록 (각 항목은 큰따옴표로 묶어주세요)
INCLUDE_PATTERNS=(
    # "re:.*pts_bbox_head.transformer.decoder.layers.0.*" #0
    # "re:.*pts_bbox_head.transformer.decoder.layers.1.*" #1 
    # "re:.*pts_bbox_head.transformer.decoder.layers.2.*" #2 
    # "re:.*pts_bbox_head.transformer.decoder.layers.3.*" #3 
    # "re:.*pts_bbox_head.transformer.decoder.layers.4.*" #4 
    # "re:.*pts_bbox_head.transformer.decoder.layers.5.*" #5 
    # "re:.*img_backbone.conv1.*" #6
    # "re:.*img_backbone.layer1.*" #7
    # "re:.*img_backbone.layer2.*" #8
    "re:.*img_backbone.layer3.*" #9
    "re:.*img_backbone.layer4.*" #10
    "re:.*img_neck.lateral_convs.*" #11
    "re:.*img_neck.fpn_convs.*" #12
    "re:.*pts_bbox_head.cls_branches.0.*" #13
    "re:.*pts_bbox_head.cls_branches.1.*" #14
    "re:.*pts_bbox_head.cls_branches.2.*" #15
    "re:.*pts_bbox_head.cls_branches.3.*" #16
    "re:.*pts_bbox_head.cls_branches.4.*" #17
    "re:.*pts_bbox_head.cls_branches.5.*" #18
    "re:.*pts_bbox_head.reg_branches.0.*" #19 
    "re:.*pts_bbox_head.reg_branches.1.*" #20
    "re:.*pts_bbox_head.reg_branches.2.*" #21
    "re:.*pts_bbox_head.reg_branches.3.*" #22
    "re:.*pts_bbox_head.reg_branches.4.*" #23
    "re:.*pts_bbox_head.reg_branches.5.*" #24
    "re:.*pts_bbox_head.transformer.encoder.layers.0.*" #25
    "re:.*pts_bbox_head.transformer.encoder.layers.1.*" #26
    "re:.*pts_bbox_head.transformer.encoder.layers.2.*" #27
    "re:.*pts_bbox_head.transformer.encoder.layers.3.*" #28
    "re:.*pts_bbox_head.transformer.encoder.layers.4.*" #29
    "re:.*pts_bbox_head.transformer.encoder.layers.5.*" #30
    "re:.*pts_bbox_head.transformer.reference_points.*" #31
    "re:.*pts_bbox_head.transformer.can_bus_mlp.*" #32
)

# NUMBER 초기화
NUMBER=9

# --- 메인 루프 ---
for INCLUDE_VALUE in "${INCLUDE_PATTERNS[@]}"; do
    PREFIX_VALUE="test_${NUMBER}"

    echo "===================================================================="
    echo "실행: NUMBER=${NUMBER}, PREFIX='${PREFIX_VALUE}'"
    echo "INCLUDE='${INCLUDE_VALUE}'"
    echo "===================================================================="

    # 첫 번째 명령어 실행
    echo "INFO: 첫 번째 명령어 실행 중..."
    PYTHONPATH=$(pwd):$PYTHONPATH python ./tools/test.py \
        "${CONFIG_FILE}" \
        "${CHECKPOINT_FILE}" \
        --eval bbox \
        --prefix "${PREFIX_VALUE}" \
        --include "${INCLUDE_VALUE}" \
        --launcher none
    
    CMD1_EXIT_CODE=$? # 첫 번째 명령어 종료 코드 저장
    if [ ${CMD1_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: 첫 번째 명령어 실행 실패 (종료 코드: ${CMD1_EXIT_CODE})"
    else
        echo "INFO: 첫 번째 명령어 실행 완료."
    fi
    echo "--------------------------------------------------------------------"

    # 두 번째 명령어 실행
    echo "INFO: 두 번째 명령어 실행 중..."
    ./tools/dist_test.sh \
        "${CONFIG_FILE}" \
        "${CHECKPOINT_FILE}" \
        ${GPUS} \
        --prefix "${PREFIX_VALUE}"

    CMD2_EXIT_CODE=$? # 두 번째 명령어 종료 코드 저장
    if [ ${CMD2_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: 두 번째 명령어 실행 실패 (종료 코드: ${CMD2_EXIT_CODE})"
    else
        echo "INFO: 두 번째 명령어 실행 완료."
    fi
    echo "====================================================================\n"
    
    # NUMBER 증가
    NUMBER=$((NUMBER + 1))
done

echo "모든 작업이 완료되었습니다."