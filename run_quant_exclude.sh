#!/bin/bash

# --- 스크립트 설정 ---
CONFIG_FILE="./projects/configs/bevformer/bevformer_base.py"
CHECKPOINT_FILE="./ckpts/bevformer_r101_dcn_24ep.pth"

# EXCLUDE_PATTERNS 목록 (예시: 15개 항목)
# 실제 사용 시 이 부분을 채워주세요. 주석에는 23개로 되어 있었으나,
# 현재 로직은 배열의 실제 크기를 기반으로 동적으로 분배합니다.
# OFFSET=33
# EXCLUDE_PATTERNS=(
#     "re:.*pts_bbox_head.transformer.can_bus_mlp.*"
#     "re:.*pts_bbox_head.transformer.reference_points.*"
#     "re:.*pts_bbox_head.transformer.can_bus_mlp.* , .*pts_bbox_head.transformer.reference_points.*"
# )
OFFSET=36
EXCLUDE_PATTERNS=(
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , re:.*pts_bbox_head.transformer.reference_points.* "   #35 번 regex 잘못써서 재실험
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , re:.*pts_bbox_head.transformer.reference_points.* ,  re:.*img_backbone.layer1.0.* "
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , re:.*pts_bbox_head.transformer.reference_points.* ,  re:.*img_backbone.layer1.1.* "
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , re:.*pts_bbox_head.transformer.reference_points.* ,  re:.*img_backbone.layer1.2.* "
    "re:.*pts_bbox_head.transformer.can_bus_mlp.* , re:.*pts_bbox_head.transformer.reference_points.* ,  re:.*img_backbone.layer1.* "
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

# --- 메인 루프 준비 ---
total_patterns=${#EXCLUDE_PATTERNS[@]}
num_gpus=8 # 고정된 GPU 수 (DEVICE_ID 0-7)

if [ ${total_patterns} -eq 0 ]; then
    echo "INFO: EXCLUDE_PATTERNS 배열이 비어있어 처리할 패턴이 없습니다."
    # 스크립트를 여기서 종료하거나, 필요에 따라 다른 처리를 할 수 있습니다.
    # exit 0 또는 아래 로직에서 my_num_patterns_to_process가 0이 되어 루프 안 탐
fi

# 각 GPU가 기본적으로 처리할 패턴 수
base_patterns_per_gpu=$((total_patterns / num_gpus))
# 나머지 패턴 수 (이 수만큼의 GPU가 1개씩 더 처리)
remainder_patterns=$((total_patterns % num_gpus))

# 현재 DEVICE_ID가 처리할 패턴 수와 시작 인덱스 계산
my_num_patterns_to_process=0
my_start_index=0

# 현재 DEVICE_ID가 나머지 패턴을 처리해야 하는지 여부 확인
if [ "${DEVICE_ID}" -lt "${remainder_patterns}" ]; then
    # 나머지 패턴을 할당받는 GPU (base_patterns_per_gpu + 1 개 처리)
    my_num_patterns_to_process=$((base_patterns_per_gpu + 1))
    # 이 GPU 이전의 GPU들은 모두 (base_patterns_per_gpu + 1)개씩 처리했음
    my_start_index=$((DEVICE_ID * (base_patterns_per_gpu + 1)))
else
    # 기본 패턴 수만 처리하는 GPU (base_patterns_per_gpu 개 처리)
    my_num_patterns_to_process=$((base_patterns_per_gpu))
    # 이 GPU 이전에는 remainder_patterns 개의 GPU가 (base_patterns_per_gpu + 1)개씩 처리했고,
    # (DEVICE_ID - remainder_patterns) 개의 GPU가 base_patterns_per_gpu개씩 처리했음
    my_start_index=$((remainder_patterns * (base_patterns_per_gpu + 1) + (DEVICE_ID - remainder_patterns) * base_patterns_per_gpu))
fi

echo "총 패턴 수 (total_patterns): ${total_patterns}"
echo "총 GPU 수 (num_gpus): ${num_gpus}"
echo "GPU당 기본 할당 패턴 수 (base_patterns_per_gpu): ${base_patterns_per_gpu}"
echo "추가 할당이 필요한 GPU 수 (remainder_patterns): ${remainder_patterns}"
echo "--------------------------------------------------------------------"
echo "DEVICE_ID ${DEVICE_ID}의 처리 시작 인덱스 (my_start_index): ${my_start_index}"
echo "DEVICE_ID ${DEVICE_ID}의 처리할 패턴 수 (my_num_patterns_to_process): ${my_num_patterns_to_process}"
echo "===================================================================="

processed_count=0
# DEVICE_ID에 따라 계산된 만큼만 루프 실행
for (( i=0; i<my_num_patterns_to_process; i++ )); do
    # NUMBER는 EXCLUDE_PATTERNS 배열의 실제 인덱스 값임
    NUMBER=$((my_start_index + i))

    # 현재 인덱스(NUMBER)가 유효한 범위 내에 있는지 확인 (이론상 my_num_patterns_to_process가 0이 아닌 이상 항상 유효)
    if [[ ${NUMBER} -ge ${total_patterns} ]] && [[ ${my_num_patterns_to_process} -gt 0 ]]; then
        echo "CRITICAL INFO: 인덱스 ${NUMBER}는 패턴 목록 범위를 벗어납니다. 로직 오류 가능성이 있습니다. 루프를 중단합니다."
        break # 더 이상 처리할 패턴이 없음
    fi

    EXCLUDE_VALUE="${EXCLUDE_PATTERNS[NUMBER]}"
    
    # PREFIX_VALUE 생성 시 NUMBER (0기반 인덱스)에 OFFSET을 더함
    PREFIX_NUMBER_OFFSETTED=$((NUMBER + OFFSET))
    PREFIX_VALUE="test_${PREFIX_NUMBER_OFFSETTED}"

    echo "실행: 실제 배열 인덱스(NUMBER)=${NUMBER}, PREFIX='${PREFIX_VALUE}' "
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
        --exclude "${EXCLUDE_VALUE}" \
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
    echo "INFO: DEVICE_ID ${DEVICE_ID}에 대해 처리할 패턴이 없습니다 (계산된 처리 패턴 수: ${my_num_patterns_to_process})."
fi

echo "모든 지정된 작업이 완료되었습니다."