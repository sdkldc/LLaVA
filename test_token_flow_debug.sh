#!/bin/bash

# 토큰 전달 디버깅 테스트 스크립트
# Two-stage forward 파이프라인에서 토큰이 의도대로 전달되는지 확인

echo "=================================="
echo "토큰 전달 디버깅 테스트"
echo "=================================="
echo ""

# GPU 선택 (기본값: 0)
export CUDA_VISIBLE_DEVICES=${1:-0}

echo "🔧 설정:"
echo "  - GPU: $CUDA_VISIBLE_DEVICES"
echo "  - 모델: checkpoints/1-step-test"
echo "  - 데이터: 최소 샘플 (빠른 테스트)"
echo ""

# 테스트용 최소 설정
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path playground/data/llava_v1_5_mix665k.json \
    --image_folder playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter checkpoints/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./test_outputs/token_flow_debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
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
    --report_to none \
    --max_steps 10 \
    --use_summary_tokens True \
    --num_summary_tokens 8 \
    --kmeans_init True \
    --kmeans_metric cosine \
    --kmeans_n_iter 3 \
    --kmeans_apply_point before_projector \
    --kmeans_use_nearest True \
    --use_dual_lora False \
    2>&1 | tee test_outputs/token_flow_debug.log

echo ""
echo "=================================="
echo "✅ 테스트 완료!"
echo "=================================="
echo ""
echo "📋 로그 확인:"
echo "  grep '토큰전달' test_outputs/token_flow_debug.log"
echo ""
echo "🔍 각 단계별 확인:"
echo "  grep '토큰전달-1' test_outputs/token_flow_debug.log  # 입력 이미지"
echo "  grep '토큰전달-2' test_outputs/token_flow_debug.log  # 1st forward 입력"
echo "  grep '토큰전달-3' test_outputs/token_flow_debug.log  # 대표 토큰 hidden states"
echo "  grep '토큰전달-4' test_outputs/token_flow_debug.log  # 2nd forward 준비"
echo "  grep '토큰전달-5' test_outputs/token_flow_debug.log  # 2nd forward 입력"
echo "  grep '토큰전달-6' test_outputs/token_flow_debug.log  # 2nd forward 출력"
echo "  grep '토큰전달-7' test_outputs/token_flow_debug.log  # Loss"
echo ""
