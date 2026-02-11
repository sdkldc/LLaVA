#!/bin/bash
# Two-stage Inference: VQAv2 Evaluation

export CUDA_VISIBLE_DEVICES="3"

gpu_list="${CUDA_VISIBLE_DEVICES:-3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="${CKPT:-llava-v1.5-7b-token-compress}"
SPLIT="${SPLIT:-llava_vqav2_mscoco_test-dev2015}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/llava-v1.5-7b-token-compress \
        --model-base ./checkpoints/llava-v1.5-7b \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use-summary-tokens &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

# Usage:
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/compression/eval/vqav2.sh
# Submit to: https://eval.ai/web/challenges/challenge-page/830/my-submission
