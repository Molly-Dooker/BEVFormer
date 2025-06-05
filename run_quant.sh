#!/bin/bash

# --- 스크립트 설정 ---
CONFIG_FILE="./projects/configs/bevformer/bevformer_base.py"
CHECKPOINT_FILE="./ckpts/bevformer_r101_dcn_24ep.pth"

# INCLUDE_PATTERNS 목록 (총 23개 항목, 인덱스 0-22)
INCLUDE_PATTERNS=(
    "re:.*img_backbone.layer4.*"                 # 인덱스 0  -> PREFIX_VALUE test_10
    "re:.*img_neck.lateral_convs.*"              # 인덱스 1  -> PREFIX_VALUE test_11
    "re:.*img_neck.fpn_convs.*"                  # 인덱스 2  -> PREFIX_VALUE test_12
    "re:.*pts_bbox_head.cls_branches.0.*"        # 인덱스 3  -> PREFIX_VALUE test_13
    "re:.*pts_bbox_head.cls_branches.1.*"        # 인덱스 4  -> PREFIX_VALUE test_14
    "re:.*pts_bbox_head.cls_branches.2.*"        # 인덱스 5  -> PREFIX_VALUE test_15
    "re:.*pts_bbox_head.cls_branches.3.*"        # 인덱스 6  -> PREFIX_VALUE test_16
    "re:.*pts_bbox_head.cls_branches.4.*"        # 인덱스 7  -> PREFIX_VALUE test_17
    "re:.*pts_bbox_head.cls_branches.5.*"        # 인덱스 8  -> PREFIX_VALUE test_18
    "re:.*pts_bbox_head.reg_branches.0.*"        # 인덱스 9  -> PREFIX_VALUE test_19
    "re:.*pts_bbox_head.reg_branches.1.*"        # 인덱스 10 -> PREFIX_VALUE test_20
    "re:.*pts_bbox_head.reg_branches.2.*"        # 인덱스 11 -> PREFIX_VALUE test_21
    "re:.*pts_bbox_head.reg_branches.3.*"        # 인덱스 12 -> PREFIX_VALUE test_22
    "re:.*pts_bbox_head.reg_branches.4.*"        # 인덱스 13 -> PREFIX_VALUE test_23
    "re:.*pts_bbox_head.reg_branches.5.*"        # 인덱스 14 -> PREFIX_VALUE test_24
    "re:.*pts_bbox_head.transformer.encoder.layers.0.*" # 인덱스 15 -> PREFIX_VALUE test_25
    "re:.*pts_bbox_head.transformer.encoder.layers.1.*" # 인덱스 16 -> PREFIX_VALUE test_26
    "re:.*pts_bbox_head.transformer.encoder.layers.2.*" # 인덱스 17 -> PREFIX_VALUE test_27
    "re:.*pts_bbox_head.transformer.encoder.layers.3.*" # 인덱스 18 -> PREFIX_VALUE test_28
    "re:.*pts_bbox_head.transformer.encoder.layers.4.*" # 인덱스 19 -> PREFIX_VALUE test_29
    "re:.*pts_bbox_head.transformer.encoder.layers.5.*" # 인덱스 20 -> PREFIX_VALUE test_30
    "re:.*pts_bbox_head.transformer.reference_points.*" # 인덱스 21 -> PREFIX_VALUE test_31
    "re:.*pts_bbox_head.transformer.can_bus_mlp.*"      # 인덱스 22 -> PREFIX_VALUE test_32
)

# --- 스크립트 입력 인자 (DEVICE_ID) 처리 ---
if [ -z "$1" ]; then
    echo "사용법: $0 <DEVICE_ID>"
    echo "  <DEVICE_ID>: 0 부터 7 까지의 숫자."
    exit 1
fi

DEVICE_ID=$1

if ! [[ "$DEVICE_ID" =~ ^[0-7]$ ]]; then
    echo "오류: DEVICE_ID는 0에서 7 사이의 정수여야 합니다."
    exit 1
fi

echo "선택된 DEVICE_ID: ${DEVICE_ID}"
echo "CUDA_VISIBLE_DEVICES가 ${DEVICE_ID}로 설정됩니다."
echo "--------------------------------------------------------------------"

# --- 메인 루프 ---
# DEVICE_ID에 따라 처리할 INCLUDE_PATTERNS의 시작 인덱스 계산
start_array_index=$((DEVICE_ID * 3))
# 각 DEVICE_ID 당 최대 3개의 항목을 처리
max_iterations_per_device=3
total_patterns=${#INCLUDE_PATTERNS[@]}

echo "처리할 패턴 범위:"
echo "시작 인덱스 (계산값): ${start_array_index}"
echo "최대 ${max_iterations_per_device}개 항목 처리 시도 (배열 범위 내에서)"
echo "===================================================================="

processed_count=0
for (( i=0; i<max_iterations_per_device; i++ )); do
    # NUMBER는 INCLUDE_PATTERNS 배열의 실제 인덱스 값임
    NUMBER=$((start_array_index + i))

    # 현재 인덱스(NUMBER)가 유효한 범위 내에 있는지 확인
    if [[ ${NUMBER} -ge ${total_patterns} ]]; then
        echo "INFO: 인덱스 ${NUMBER}는 패턴 목록 범위를 벗어나므로 루프를 종료합니다."
        break # 더 이상 처리할 패턴이 없음
    fi

    INCLUDE_VALUE="${INCLUDE_PATTERNS[NUMBER]}"
    
    # PREFIX_VALUE 생성 시 NUMBER (0기반 인덱스)에 10을 더함
    PREFIX_NUMBER_OFFSETTED=$((NUMBER + 10))
    PREFIX_VALUE="test_${PREFIX_NUMBER_OFFSETTED}"

    echo "실행: 실제 배열 인덱스(NUMBER)=${NUMBER}, PREFIX='${PREFIX_VALUE}' (test_$((${NUMBER}+10)))"
    echo "INCLUDE='${INCLUDE_VALUE}'"
    echo "CUDA_VISIBLE_DEVICES=${DEVICE_ID}"
    echo "--------------------------------------------------------------------"

    # 명령어 실행
    echo "INFO: 명령어 실행 중..."
    
    env CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
    PYTHONPATH=$(pwd):$PYTHONPATH python ./tools/test.py \
        "${CONFIG_FILE}" \
        "${CHECKPOINT_FILE}" \
        --eval bbox \
        --prefix "${PREFIX_VALUE}" \
        --include "${INCLUDE_VALUE}" \
        --launcher none
    
    CMD_EXIT_CODE=$? 
    if [ ${CMD_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: 명령어 실행 실패 (종료 코드: ${CMD_EXIT_CODE})"
    else
        echo "INFO: 명령어 실행 완료."
    fi
    echo "====================================================================\n"
    processed_count=$((processed_count + 1))
done

if [[ ${processed_count} -eq 0 ]]; then
    echo "INFO: DEVICE_ID ${DEVICE_ID}에 대해 처리할 패턴이 없습니다 (계산된 시작 인덱스: ${start_array_index})."
fi

echo "모든 지정된 작업이 완료되었습니다."