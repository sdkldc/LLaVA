#!/bin/bash
# 이미 학습된 LLaVA v1.5 모델에 LoRA를 적용
# bash ./scripts/compression/finetune_token_compress_deepspeed_batch256.sh
# Visual Instruction Tuning(2단계 SFT) 과정에서 LoRA 적용
# 이미지 토큰 압축: 요약 토큰을 사용하여 이미지 정보 압축
# K-means 동적 초기화: 매 forward마다 이미지에서 대표 토큰 추출 (학습 안 함)
# Eager attention 사용으로 메모리 최적화 설정 적용
# --include localhost:1 : GPU 1에서 실행
# --include localhost:1,3 : GPU 1,3에서 실행
# PYTHONNOUSERSITE=1 PYTHONPATH= bash ./scripts/compression/finetune_token_compress_check.sh

# torch 2.1.x 환경에서 expandable_segments는 allocator internal assert를 유발할 수 있음
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

deepspeed --include localhost:1,3 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True --lora_r 64 --lora_alpha 128 \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.filtered.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-check64 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_steps 1000 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_summary_tokens True \
    --num_summary_tokens 32 \
    --tune_summary_tokens False \
    --use_dual_lora True \
    --kmeans_init True \
    --kmeans_metric cosine \
    --kmeans_n_iter 3 \
    --kmeans_apply_point before_projector \
    --kmeans_use_nearest True 

# Eager attention 메모리 최적화 설정:
# - gradient_accumulation_steps: 64 -> 256 (4배 증가)
# - Effective batch size = 4 * 256 = 1024 (원래 16 * 64 = 1024와 동일)

# per_device_train_batch_size: GPU1개당 학습 배치 크기: 공식 레포는 8개 GPU를 이용하여 16:: 16x8=128
# per_device_eval_batch_size: GPU1개당 평가 배치 크기 : 공식 레포는 8개 GPU를 이용하여 4:: 4x8=32
# output_dir: 체크포인트 저장 경로인데, 동일한 체크포인트가 있을 경우 자동으로 이어 받아 학습한다.


# lora_enable: LoRA 적용 여부, lora_r: LoRA rank, lora_alpha: LoRA alpha
# mm_projector_lr: 멀티모달 프로젝터 학습률아니면
# deepspeed: DeepSpeed 설정 파일 경로
# model_name_or_path: 사전학습된 모델 경로
# data_path: 학습 데이터 경로
# image_folder: 이미지 폴더 경로
# vision_tower: 비전 백본 모델
# mm_projector_type: 멀티모달 프로젝터 타입
# mm_vision_select_layer: 비전 백본에서 특징 추출할 레이어
# mm_use_im_start_end: 이미지 시작/끝 토큰 사용 여부
# mm_use_im_patch_token: 이미지 패치 토큰 사용 여부
# image_aspect_ratio: 이미지 종횡비 조정 방식 (pad: 패딩)
# group_by_modality_length: 모달리티 길이별로 그룹화
# bf16: bfloat16 정밀도 사용 여부
# output_dir: 체크포인트 저장 경로, 학습된 가중치
# num_train_epochs: 학습 에폭 수
# per_device_train_batch_size: 학습 배치 크기
# per_device_eval_batch_size: 평가 배치 크기
# gradient_accumulation_steps: 그래디언트 누적 스텝 수
# evaluation_strategy: 평가 전략
# save_strategy: 체크포인트 저장 전략
# save_steps: 체크포인트 저장 스텝 수
# save_total_limit: 저장할 체크포인트 최대 개수
# learning_rate: 학습률
# weight_decay: 가중치 감쇠
# warmup_ratio: 워밍업 비율
# lr_scheduler_type: 학습률 스케줄러 타입
# logging_steps: 로깅 스텝 수
# tf32: TF32 정밀도 사용 여부
# model_max_length: 모델 최대 입력 길이
# gradient_checkpointing: 그래디언트 체크포인팅 사용 여부
# dataloader_num_workers: 데이터로더 워커 수
# lazy_preprocess: 지연 전처리 사용 여부
# report_to: 로깅 플랫폼
# use_summary_tokens: 이미지 요약 토큰 사용 여부 (True/False)
# num_summary_tokens: 요약 토큰 개수 (1, 4, 8, 16, 32 등)
# tune_summary_tokens: 요약 토큰 학습 여부 (kmeans_init=False일 때만 유효, True: 학습, False: 고정)
# use_dual_lora: 1st/2nd forward에 별도 LoRA 사용 여부 (True: 2개 LoRA, False: False: 단일 LoRA)
# kmeans_init: K-means 동적 초기화 여부
#   - True: 매 forward마다 이미지에서 K-means로 대표 토큰 추출 (학습 안 함, 유의미한 초기값 제공)
#   - False: 학습 가능한 요약 토큰 사용 (Xavier 초기화)
# kmeans_metric: K-means 거리 메트릭 (cosine: 코사인 유사도, l2: 유클리드 거리, dot: 내적)
# kmeans_n_iter: K-means 반복 횟수 (centroid 업데이트 반복 횟수, 기본값: 3)
# kmeans_apply_point: K-means 적용 시점 (기본값: before_projector)
#   - before_projector: Vision Encoder → K-means → Centroid → Projector (기본, 추천)
#   - after_projector: Vision Encoder → Projector → K-means → Centroid
# kmeans_use_nearest: 대표 토큰 선택 방식 (기본값: True)
#   - True: centroid와 가장 가까운 실제 이미지 토큰 사용 (원본 분포 보존)
#   - False: centroid 직접 사용 (평균값, 더 부드러운 표현)
