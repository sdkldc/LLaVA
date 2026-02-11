#!/usr/bin/env python3
"""Stage 2 입력 준비 테스트"""

import torch
from transformers import AutoTokenizer

# Constants
IMAGE_TOKEN_INDEX = -200

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/llava-v1.5-7b')

# 프롬프트
prompt = "<image>\nDescribe this image in detail."
print(f"Original prompt: {prompt}")

# 토크나이즈
tokens = tokenizer(prompt, return_tensors='pt')
input_ids = tokens.input_ids

print(f"\nTokenized input_ids shape: {input_ids.shape}")
print(f"Tokens: {input_ids}")
print(f"Decoded: {tokenizer.decode(input_ids[0])}")

# <image> 토큰을 IMAGE_TOKEN_INDEX로 교체
# LLaVA는 <image>를 IMAGE_TOKEN_INDEX로 교체합니다
image_token_id = tokenizer.convert_tokens_to_ids('<image>')
print(f"\n<image> token ID: {image_token_id}")

# IMAGE_TOKEN_INDEX로 교체
input_ids_with_image_token = input_ids.clone()
input_ids_with_image_token[input_ids == image_token_id] = IMAGE_TOKEN_INDEX

print(f"\nAfter replacing <image> with IMAGE_TOKEN_INDEX:")
print(f"input_ids: {input_ids_with_image_token}")
print(f"IMAGE_TOKEN_INDEX count: {(input_ids_with_image_token == IMAGE_TOKEN_INDEX).sum()}")

# 이미지 토큰 위치 찾기
image_positions = torch.where(input_ids_with_image_token[0] == IMAGE_TOKEN_INDEX)[0]
print(f"IMAGE_TOKEN positions: {image_positions.tolist()}")

# 텍스트 부분 분리 (이미지 토큰 제외)
if len(image_positions) > 0:
    image_token_indices = [-1] + image_positions.tolist() + [input_ids_with_image_token.shape[1]]
    print(f"\nimage_token_indices: {image_token_indices}")
    
    text_parts = []
    for i in range(len(image_token_indices) - 1):
        start = image_token_indices[i] + 1
        end = image_token_indices[i+1]
        text_part = input_ids_with_image_token[0, start:end]
        text_parts.append(text_part)
        print(f"Text part {i}: tokens[{start}:{end}] = {text_part.tolist()}")
        print(f"  Decoded: '{tokenizer.decode(text_part)}'")
    
    # 요약 토큰이 8개라고 가정
    num_summary_tokens = 8
    
    # Stage 2 시퀀스 길이 계산
    total_text_tokens = sum(len(part) for part in text_parts)
    stage2_length = total_text_tokens + num_summary_tokens
    
    print(f"\nStage 2 sequence composition:")
    print(f"  Text tokens: {total_text_tokens}")
    print(f"  Summary tokens: {num_summary_tokens}")
    print(f"  Total Stage 2 length: {stage2_length}")
    
    # 원본 입력과 비교
    original_length = input_ids.shape[1]
    print(f"\nComparison:")
    print(f"  Original input length: {original_length}")
    print(f"  Stage 2 input length: {stage2_length}")
    print(f"  Difference: {stage2_length - original_length}")
