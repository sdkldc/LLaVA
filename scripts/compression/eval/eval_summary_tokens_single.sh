#!/bin/bash
# 단일 이미지에 대한 Two-stage Inference 평가
# Summary tokens를 사용한 추론과 일반 추론 비교

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 기본 설정
MODEL_PATH=${1:-"./checkpoints/llava-v1.5-7b"}
IMAGE_PATH=${2:-"https://llava-vl.github.io/static/images/view.jpg"}
PROMPT=${3:-"Describe this image in detail."}

echo "=========================================="
echo "Two-stage Inference Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Image: $IMAGE_PATH"
echo "Prompt: $PROMPT"
echo "=========================================="

# Standard Inference
echo ""
echo "[1/3] Running Standard Inference (Full Image)..."
python -m llava.eval.run_llava \
    --model-path "$MODEL_PATH" \
    --image-file "$IMAGE_PATH" \
    --query "$PROMPT" \
    2>&1 | tee standard_output.log

echo ""
echo "=========================================="

# Two-stage Inference  
echo ""
echo "[2/3] Running Two-stage Inference (Summary Tokens)..."
python test_inference_summary_tokens.py \
    --model-path "$MODEL_PATH" \
    --image "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --use-summary-tokens \
    2>&1 | tee twostage_output.log

echo ""
echo "=========================================="

# Comparison
echo ""
echo "[3/3] Comparing Both Methods..."
python test_inference_summary_tokens.py \
    --model-path "$MODEL_PATH" \
    --image "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --compare \
    2>&1 | tee comparison_output.log

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to:"
echo "  - standard_output.log"
echo "  - twostage_output.log"
echo "  - comparison_output.log"
echo "=========================================="
