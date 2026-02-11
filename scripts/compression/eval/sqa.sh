#!/bin/bash
# Two-stage Inference: ScienceQA Evaluation

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b-token-compress \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-token-compress.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-token-compress.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-token-compress_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-token-compress_result.json

# Usage:
# CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/sqa.sh
