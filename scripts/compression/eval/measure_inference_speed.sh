#!/bin/bash
# Inference 속도 측정
# Standard vs Two-stage 속도 비교

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${1:-"./checkpoints/llava-v1.5-7b-summary-tokens-lora"}
NUM_SAMPLES=${2:-100}
OUTPUT_FILE="inference_speed_results.txt"

echo "=========================================="
echo "Inference Speed Measurement"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Python 스크립트로 속도 측정
python << EOF
import torch
import time
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
from io import BytesIO

# 모델 로드
print("Loading model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="$MODEL_PATH",
    model_base=None,
    model_name=get_model_name_from_path("$MODEL_PATH"),
    device_map='auto'
)
print(f"Model loaded: {model.config.model_type}")

# 테스트 이미지 로드
image_url = "https://llava-vl.github.io/static/images/view.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')
image_size = image.size

# 이미지 전처리
image_tensor = process_images([image], image_processor, model.config)
if isinstance(image_tensor, list):
    image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

# 프롬프트 준비
conv = conv_templates["llava_v1"].copy()
prompt = DEFAULT_IMAGE_TOKEN + '\\n' + "Describe this image."
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(model.device)

num_samples = $NUM_SAMPLES
max_new_tokens = 128

# Warmup
print("\\nWarming up...")
for _ in range(5):
    with torch.inference_mode():
        _ = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_summary_tokens=False,
            max_new_tokens=32,
            do_sample=False,
        )
torch.cuda.empty_cache()

# Standard Inference 속도 측정
print(f"\\n[1/2] Measuring Standard Inference ({num_samples} samples)...")
standard_times = []
for i in range(num_samples):
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.inference_mode():
        _ = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_summary_tokens=False,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    standard_times.append(elapsed)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{num_samples}")

# Two-stage Inference 속도 측정
print(f"\\n[2/2] Measuring Two-stage Inference ({num_samples} samples)...")
twostage_times = []
for i in range(num_samples):
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.inference_mode():
        _ = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_summary_tokens=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    twostage_times.append(elapsed)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{num_samples}")

# 결과 분석
standard_mean = np.mean(standard_times)
standard_std = np.std(standard_times)
twostage_mean = np.mean(twostage_times)
twostage_std = np.std(twostage_times)

speedup = standard_mean / twostage_mean

# 결과 출력 및 저장
results = f"""
========================================
Inference Speed Comparison Results
========================================

Configuration:
  - Model: $MODEL_PATH
  - Samples: {num_samples}
  - Max new tokens: {max_new_tokens}
  - Device: {model.device}

Standard Inference:
  - Mean: {standard_mean:.4f} sec
  - Std:  {standard_std:.4f} sec
  - Min:  {min(standard_times):.4f} sec
  - Max:  {max(standard_times):.4f} sec

Two-stage Inference:
  - Mean: {twostage_mean:.4f} sec
  - Std:  {twostage_std:.4f} sec
  - Min:  {min(twostage_times):.4f} sec
  - Max:  {max(twostage_times):.4f} sec

Performance:
  - Speedup: {speedup:.2f}x
  - Time reduction: {(1 - twostage_mean/standard_mean)*100:.1f}%

========================================
"""

print(results)

# 파일 저장
with open("$OUTPUT_FILE", "w") as f:
    f.write(results)

print(f"Results saved to: $OUTPUT_FILE")
EOF

echo ""
echo "=========================================="
echo "Speed measurement completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="
