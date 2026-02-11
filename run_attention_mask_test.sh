#!/bin/bash

# Attention Mask 디버깅 테스트 실행 스크립트

echo "=================================="
echo "Attention Mask 디버깅 테스트"
echo "=================================="
echo ""

# 1. 기본 마스크 함수 테스트
echo "[1/2] 기본 Attention Mask 함수 테스트 실행..."
echo "--------------------------------------"
python test_attention_mask_debug.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 기본 마스크 테스트 실패"
    exit 1
fi

echo ""
echo "✅ 기본 마스크 테스트 완료"
echo ""
echo "생성된 파일:"
echo "  - attention_mask_stage_comparison.png"
echo "  - attention_mask_detailed.png"
echo ""

# 2. 실제 모델 적용 테스트
echo "[2/2] 실제 모델 Attention Mask 적용 테스트 실행..."
echo "--------------------------------------"
python test_attention_mask_model.py

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ 모델 테스트에서 일부 오류 발생 (정상일 수 있음)"
    echo "   - 모델 파일이 없거나 summary_tokens 모듈이 비활성화된 경우"
else
    echo ""
    echo "✅ 모델 테스트 완료"
fi

echo ""
echo "=================================="
echo "전체 테스트 완료"
echo "=================================="
echo ""
echo "📊 결과 확인:"
echo "  1. 기본 마스크 함수 테스트 로그 확인"
echo "  2. 시각화 이미지 확인 (png 파일)"
echo "  3. 모델 적용 테스트 로그 확인"
echo ""
echo "✅ 의도한 동작:"
echo "  - Stage 1: Causal mask + 요약 토큰끼리 참조 불가"
echo "  - Stage 2: Causal mask + Padding mask (요약 토큰끼리 참조 가능)"
echo ""
