#!/bin/bash
# Feature alignment가 완료된 모델에 LoRA를 적용하여 visual instruction tuning을 수행.
# 2번째 학습에 해당한다 
deepspeed llava/train/train_mem.py \

    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
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
    --report_to wandb


    # lora_enable: LoRA 적용 여부, lora_r: LoRA rank, lora_alpha: LoRA alpha
    # mm_projector_lr: 멀티모달 프로젝터 학습률

    # deepspeed: DeepSpeed 설정 파일 경로
    # model_name_or_path: 사전학습된 모델 경로
    # data_path: 학습 데이터 경로
    # image_folder: 이미지 폴더 경로
    # vision_tower: 비전 백본 모델
    # pretrain_mm_mlp_adapter: 사전학습된 멀티모달 어댑터 경로
    # mm_projector_type: 멀티모달 프로젝터 타입
    # mm_vision_select_layer: 비전 백본에서 특징 추출할 레이어
    # mm_use_im_start_end: 이미지 시작/끝 토큰 사용 여부
    # mm_use_im_patch_token: 이미지 패치 토큰 사용 여부
    # image_aspect_ratio: 이미지 종횡비 조정 방식 (pad: 패딩)
    # group_by_modality_length: 모달리티 길이별로 그룹화
    # bf16: bfloat16 정밀도 사용 여부
    # output_dir: 체크포인트 저장 경로
    # num_train_epochs: 학습 에폭 수
    # per_device_train_batch_size: 학습 배치 크기
    # per_device_eval_batch_size: 평가 배치 크기
    # gradient_accumulation_steps: 그래디언트 누적 스텝 수
    # evaluation_strategy: 평가 전략 (no: 평가 안함)
    # save_strategy: 체크포인트 저장 전략 (steps: 일정 스텝마다 저장
