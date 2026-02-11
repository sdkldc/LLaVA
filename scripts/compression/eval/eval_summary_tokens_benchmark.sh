#!/bin/bash
# Benchmark 데이터셋에 대한 Two-stage Inference 평가
# Summary tokens의 성능과 속도 측정

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 기본 설정
MODEL_PATH=${1:-"./checkpoints/llava-v1.5-7b-summary-tokens-lora"}
BENCHMARK_NAME=${2:-"mmbench"}  # mmbench, pope, scienceqa 등
OUTPUT_DIR="./eval_results/summary_tokens"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Benchmark Evaluation with Summary Tokens"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Benchmark: $BENCHMARK_NAME"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Benchmark별 평가
case "$BENCHMARK_NAME" in
    "mmbench")
        echo ""
        echo "[MMBench Evaluation]"
        
        # Standard
        echo "Running Standard Inference..."
        python -m llava.eval.model_vqa_mmbench \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
            --answers-file "$OUTPUT_DIR/mmbench_standard.jsonl" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1
        
        # Two-stage
        echo "Running Two-stage Inference..."
        python -m llava.eval.model_vqa_mmbench \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
            --answers-file "$OUTPUT_DIR/mmbench_twostage.jsonl" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-summary-tokens
        ;;
        
    "pope")
        echo ""
        echo "[POPE Evaluation]"
        
        # Standard
        echo "Running Standard Inference..."
        python -m llava.eval.model_vqa_loader \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder ./playground/data/eval/pope/val2014 \
            --answers-file "$OUTPUT_DIR/pope_standard.jsonl" \
            --temperature 0 \
            --conv-mode vicuna_v1
        
        # Two-stage
        echo "Running Two-stage Inference..."
        python -m llava.eval.model_vqa_loader \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder ./playground/data/eval/pope/val2014 \
            --answers-file "$OUTPUT_DIR/pope_twostage.jsonl" \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-summary-tokens
        
        # Evaluate
        echo "Evaluating POPE results..."
        python llava/eval/eval_pope.py \
            --annotation-dir ./playground/data/eval/pope/coco \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --result-file "$OUTPUT_DIR/pope_standard.jsonl" > "$OUTPUT_DIR/pope_standard_results.txt"
        
        python llava/eval/eval_pope.py \
            --annotation-dir ./playground/data/eval/pope/coco \
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
            --result-file "$OUTPUT_DIR/pope_twostage.jsonl" > "$OUTPUT_DIR/pope_twostage_results.txt"
        ;;
        
    "scienceqa")
        echo ""
        echo "[ScienceQA Evaluation]"
        
        # Standard
        echo "Running Standard Inference..."
        python -m llava.eval.model_vqa_science \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder ./playground/data/eval/scienceqa/images/test \
            --answers-file "$OUTPUT_DIR/scienceqa_standard.jsonl" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1
        
        # Two-stage
        echo "Running Two-stage Inference..."
        python -m llava.eval.model_vqa_science \
            --model-path "$MODEL_PATH" \
            --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder ./playground/data/eval/scienceqa/images/test \
            --answers-file "$OUTPUT_DIR/scienceqa_twostage.jsonl" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --use-summary-tokens
        ;;
        
    *)
        echo "Unknown benchmark: $BENCHMARK_NAME"
        echo "Supported: mmbench, pope, scienceqa"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Benchmark evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
