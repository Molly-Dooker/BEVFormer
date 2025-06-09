#!/bin/bash

# 5시간 대기 (5시간 * 60분 * 60초)
echo "대기시작"
sleep 18000



for NUMBER in {33..36}
do
  echo "NUMBER ${NUMBER}에 대한 테스트를 시작합니다."
  ./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth 8 --prefix test_${NUMBER}
  echo "NUMBER ${NUMBER}에 대한 테스트를 완료했습니다."
done

echo "모든 테스트가 완료되었습니다."