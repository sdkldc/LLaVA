#!/bin/bash

# K-means 초기화 시스템 통합 테스트 스크립트
# 실제 훈련 파이프라인에서 다양한 K-means 설정을 테스트

echo "========================================================================"
echo "K-means 초기화 시스템 통합 테스트"
echo "========================================================================"

# 기본 설정
MODEL_PATH="checkpoints/llava-v1.5-7b"
DATA_PATH="playground/data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="playground/data"
OUTPUT_BASE="checkpoints/kmeans-integration-test"

# 테스트할 설정들
CONFIGS=(
    # "이름|metric|n_iter|apply_point|use_nearest"
    "test1-cosine-before-nearest|cosine|3|before_projector|True"
    "test2-cosine-before-centroid|cosine|3|before_projector|False"
    "test3-l2-before-nearest|l2|3|before_projector|True"
    "test4-cosine-after-nearest|cosine|3|after_projector|True"
    "test5-cosine-after-centroid|cosine|3|after_projector|False"
)

# 각 설정별로 1 step씩 훈련해서 초기화 확인
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME METRIC N_ITER APPLY_POINT USE_NEAREST <<< "$config"
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo "테스트: $NAME"
    echo "  - metric: $METRIC"
    echo "  - n_iter: $N_ITER"
    echo "  - apply_point: $APPLY_POINT"
    echo "  - use_nearest: $USE_NEAREST"
    echo "--------------------------------------------------------------------"
    
    OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    
    deepspeed llava/train/train_mem.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path $MODEL_PATH \
        --version v1 \
        --data_path $DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name "kmeans-test-${NAME}" \
        --lora_enable True \
        --lora_r 128 \
        --lora_alpha 256 \
        --mm_projector_lr 2e-5 \
        --num_summary_tokens 16 \
        --tune_summary_tokens False \
        --use_dual_lora True \
        --kmeans_init True \
        --kmeans_metric $METRIC \
        --kmeans_n_iter $N_ITER \
        --kmeans_apply_point $APPLY_POINT \
        --kmeans_use_nearest $USE_NEAREST \
        --max_steps 1
    
    if [ $? -eq 0 ]; then
        echo "✓ $NAME 테스트 성공"
    else
        echo "✗ $NAME 테스트 실패"
        exit 1
    fi
done

echo ""
echo "========================================================================"
echo "✅ 모든 K-means 설정 통합 테스트 완료!"
echo "========================================================================"
echo ""
echo "[테스트 완료 설정]"
echo "  ✓ cosine + before_projector + nearest"
echo "  ✓ cosine + before_projector + centroid"
echo "  ✓ l2 + before_projector + nearest"
echo "  ✓ cosine + after_projector + nearest"
echo "  ✓ cosine + after_projector + centroid"
echo ""
echo "결과 확인:"
echo "  - 각 테스트의 로그에서 'K-means initialization' 메시지 확인"
echo "  - Summary token shape 확인"
echo "  - 훈련 loss가 정상적으로 계산되는지 확인"
