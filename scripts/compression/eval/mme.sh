#!/bin/bash
# Two-stage Inference: MME Evaluation

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-token-compress-batch128 \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-token8-step1000.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-token8-step1000

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-token8-step1000
# Usage:
# CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/mme.sh

# 고정
# model-base: 기준 모델 경로
# question-file: ./playground/data/eval/MME/llava_mme.jsonl
# image-folder: ./playground/data/eval/MME/MME_Benchmark_release_version

# 변경
# model-path: 대상 모델 경로
# answers-file: 출력 답안 파일 경로
# result-file: 평가 결과 파일 경로