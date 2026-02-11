#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="${SPLIT:-llava_gqa_testdev_balanced}"
GQADIR="${GQADIR:-./playground/data/eval/gqa/data}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/llava-v1.5-7b \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

if [ ! -d "$GQADIR" ]; then
    echo "Missing GQA data dir: $GQADIR" >&2
    exit 1
fi

cd "$GQADIR" || exit 1
if [ -f "eval/eval.py" ]; then
    python eval/eval.py --tier testdev_balanced
elif [ -f "eval.py" ]; then
    python eval.py --tier testdev_balanced
else
    echo "Missing eval script: $GQADIR/eval/eval.py or $GQADIR/eval.py" >&2
    exit 1
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/gqa.sh
# 재현 성공