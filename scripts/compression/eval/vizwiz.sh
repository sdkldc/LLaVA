#!/bin/bash
# Two-stage Inference: VizWiz Evaluation

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-token-compress \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-token-compress.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-token-compress.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-token-compress.json

# Usage:
# CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/vizwiz.sh
# Submit to: https://eval.ai/auth/login (VizWiz test server)
# Upload file: ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-token-compress.json
