#!/bin/bash
# README for Compression Evaluation Scripts

## 평가 스크립트 모음

이 디렉토리는 Summary Tokens를 사용한 Two-stage Inference 평가 스크립트들을 포함합니다.

### 1. 단일 이미지 평가
```bash
# 기본 사용
bash scripts/compression/eval/eval_summary_tokens_single.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    "https://example.com/image.jpg" \
    "Describe this image in detail."

# CUDA 디바이스 지정
CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/eval_summary_tokens_single.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    ./test_image.jpg \
    "What is in this image?"
```

### 2. 벤치마크 평가
```bash
# MMBench
bash scripts/compression/eval/eval_summary_tokens_benchmark.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    mmbench

# POPE
bash scripts/compression/eval/eval_summary_tokens_benchmark.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    pope

# ScienceQA
bash scripts/compression/eval/eval_summary_tokens_benchmark.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    scienceqa
```

### 3. 속도 측정
```bash
# 100 샘플로 속도 측정
bash scripts/compression/eval/measure_inference_speed.sh \
    ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    100

# 결과 확인
cat inference_speed_results.txt
```

### 4. 품질 비교
```bash
# 데이터셋 준비 (JSON 형식)
# [
#   {
#     "image": "path/to/image1.jpg",
#     "prompt": "What is in this image?",
#     "answer": "optional ground truth"
#   },
#   ...
# ]

# 평가 실행
python scripts/compression/eval/eval_quality_comparison.py \
    --model-path ./checkpoints/llava-v1.5-7b-summary-tokens-lora \
    --dataset ./eval_dataset.json \
    --output ./quality_results.json

# 결과 분석만
python scripts/compression/eval/eval_quality_comparison.py \
    --output ./quality_results.json \
    --analyze-only
```

### 스크립트 설명

- **eval_summary_tokens_single.sh**: 단일 이미지에 대한 빠른 테스트
- **eval_summary_tokens_benchmark.sh**: 표준 벤치마크 평가
- **measure_inference_speed.sh**: 추론 속도 측정 및 비교
- **eval_quality_comparison.py**: 출력 품질 상세 비교

### 결과 파일

평가 후 생성되는 파일들:
- `standard_output.log`: Standard inference 결과
- `twostage_output.log`: Two-stage inference 결과
- `comparison_output.log`: 비교 결과
- `inference_speed_results.txt`: 속도 측정 결과
- `quality_comparison_results.json`: 품질 비교 결과
- `eval_results/summary_tokens/`: 벤치마크 결과

### 주의사항

1. 모델 경로가 올바른지 확인
2. Summary tokens 학습된 모델 사용 필요
3. `use_summary_tokens=True` 플래그가 제대로 전달되는지 확인
4. GPU 메모리 충분한지 확인

### 예상 결과

- **속도**: Two-stage가 Standard보다 빠를 수 있음 (압축 효과)
- **품질**: 두 방법 간 유사한 성능 (학습이 잘 되었다면)
- **압축률**: 576 토큰 → 8 토큰 (약 72배 압축)
