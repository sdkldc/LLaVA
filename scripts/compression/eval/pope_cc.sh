#!/bin/bash
# Two-stage Inference: POPE Evaluation
# Summary tokens + Dual LoRA로 학습된 모델 평가

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-compress_cache-batch128-token32 \
    --model-base ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-compress_cache-batch128-token32.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-summary-tokens

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-compress_cache-batch128-token32.jsonl


# model-path, answers-file, result-file 경로를 실제 실험에 맞게 수정 필요
# # CUDA_VISIBLE_DEVICES=2 bash ./scripts/compression/eval/pope_cc.sh


# 고정 1
# question-file: ./playground/data/eval/pope/llava_pope_test.jsonl
# image-folder: ./playground/data/eval/pope/val2014
# 고정 2
# annotation-dir: ./playground/data/eval/pope/coco
# question-file: ./playground/data/eval/pope/llava_pope_test.jsonl