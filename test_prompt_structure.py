#!/usr/bin/env python3
"""실제 프롬프트가 어떻게 구성되는지 확인"""

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/llava-v1.5-7b')

# 프롬프트 준비
prompt = "Describe this image in detail."
conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()

# 이미지 토큰 추가
inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt_text = conv.get_prompt()

print("=" * 80)
print("Final prompt text:")
print("=" * 80)
print(prompt_text)
print("=" * 80)

# 토크나이즈
input_ids = tokenizer_image_token(
    prompt_text, 
    tokenizer, 
    IMAGE_TOKEN_INDEX, 
    return_tensors='pt'
)

print(f"\nInput IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids}")

# IMAGE_TOKEN_INDEX 위치 찾기
image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
image_token_positions = image_token_mask.nonzero(as_tuple=True)[0]
print(f"\nIMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX}) positions: {image_token_positions.tolist()}")
print(f"Number of image tokens: {len(image_token_positions)}")

# 각 토큰 디코딩
print("\nToken breakdown:")
for i, token_id in enumerate(input_ids[0].tolist()):
    if token_id == IMAGE_TOKEN_INDEX:
        print(f"  [{i:2d}] {token_id:6d} = <IMAGE_TOKEN>")
    else:
        decoded = tokenizer.decode([token_id])
        print(f"  [{i:2d}] {token_id:6d} = '{decoded}'")
