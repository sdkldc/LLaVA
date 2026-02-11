#!/bin/bash
# Two-stage Inference: TextVQA Evaluation

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-token-compress \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-token-compress.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-token-compress.jsonl

# Usage:
# CUDA_VISIBLE_DEVICES=3 bash scripts/compression/eval/textvqa.sh
