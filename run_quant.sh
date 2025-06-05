#!/bin/bash

# --- 스크립트 설정 ---
CONFIG_FILE="./projects/configs/bevformer/bevformer_base.py"
CHECKPOINT_FILE="./ckpts/bevformer_r101_dcn_24ep.pth"

# EXCLUDE_PATTERNS 목록 (총 23개 항목, 인덱스 0-22)
OFFSET = 33
EXCLUDE_PATTERNS=(
    "re:.*pts_bbox_head.transformer.can_bus_mlp.*"
    "re:.*pts_bbox_head.transformer.reference_points.*"
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , .*pts_bbox_head.transformer.reference_points.*"
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
# DEVICE_ID에 따라 처리할 EXCLUDE_PATTERNS의 시작 인덱스 계산
start_array_index=$((DEVICE_ID * 3))
# 각 DEVICE_ID 당 최대 3개의 항목을 처리
max_iterations_per_device=3
total_patterns=${#EXCLUDE_PATTERNS[@]}

echo "처리할 패턴 범위:"
echo "시작 인덱스 (계산값): ${start_array_index}"
echo "최대 ${max_iterations_per_device}개 항목 처리 시도 (배열 범위 내에서)"
echo "===================================================================="

processed_count=0
for (( i=0; i<max_iterations_per_device; i++ )); do
    # NUMBER는 EXCLUDE_PATTERNS 배열의 실제 인덱스 값임
    NUMBER=$((start_array_index + i))

    # 현재 인덱스(NUMBER)가 유효한 범위 내에 있는지 확인
    if [[ ${NUMBER} -ge ${total_patterns} ]]; then
        echo "INFO: 인덱스 ${NUMBER}는 패턴 목록 범위를 벗어나므로 루프를 종료합니다."
        break # 더 이상 처리할 패턴이 없음
    fi

    EXCLUDE_VALUE="${EXCLUDE_PATTERNS[NUMBER]}"
    
    # PREFIX_VALUE 생성 시 NUMBER (0기반 인덱스)에 10을 더함
    PREFIX_NUMBER_OFFSETTED=$((NUMBER + ${OFFSET}))
    PREFIX_VALUE="test_${PREFIX_NUMBER_OFFSETTED}"

    echo "실행: 실제 배열 인덱스(NUMBER)=${NUMBER}, PREFIX='${PREFIX_VALUE}' (test_$((${NUMBER}+${OFFSET})))"
    echo "EXCLUDE='${EXCLUDE_VALUE}'"
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
        --EXCLUDE "${EXCLUDE_VALUE}" \
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