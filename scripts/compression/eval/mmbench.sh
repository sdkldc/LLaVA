#!/bin/bash
# Two-stage Inference: MMBench Evaluation

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-v1.5-7b-token-compress \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-token-compress.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-token-compress

# Usage:
# CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/mmbench.sh
# Upload: playground/data/eval/mmbench/answers_upload/<SPLIT>/<experiment>
# Submit to: https://mmbench.opencompass.org.cn/mmbench-submission
