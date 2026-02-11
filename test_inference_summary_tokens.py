#!/usr/bin/env python3
"""
Two-stage Inference 테스트 스크립트

요약 토큰을 사용한 추론이 정상적으로 작동하는지 확인
"""

import argparse
import os
from io import BytesIO

import requests
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model, load_pretrained_model_dual_lora
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates


def load_image(image_file):
    """이미지 로드 (URL 또는 로컬 파일)"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def test_inference_basic(model_path, image_file, prompt, use_summary_tokens=False, model_base=None):
    """
    기본 inference 테스트
    
    Args:
        model_path: 모델 경로
        image_file: 이미지 파일 경로 또는 URL
        prompt: 질문 프롬프트
        use_summary_tokens: True이면 two-stage inference 사용
    """
    print("=" * 70)
    print(f"TEST: {'Two-stage' if use_summary_tokens else 'Standard'} Inference")
    print("=" * 70)
    
    # 모델 로드
    print(f"\n[1] Loading model from: {model_path}")
    model_path = os.path.expanduser(model_path)
    use_dual_lora_loader = (
        use_summary_tokens
        and model_base is not None
        and os.path.exists(os.path.join(model_path, "adapter_config.json"))
        and os.path.exists(os.path.join(model_path, "summary_utilizer"))
    )

    if use_dual_lora_loader:
        print("Using Dual LoRA loader (default + summary_utilizer)")
        tokenizer, model, image_processor, context_len = load_pretrained_model_dual_lora(
            model_path=model_path,
            model_base=model_base,
            model_name=get_model_name_from_path(model_path),
            device_map='auto'
        )
    else:
        if use_summary_tokens and model_base is None:
            print("Warning: --model-base is not set, falling back to standard loader")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=get_model_name_from_path(model_path),
            device_map='auto'
        )

    # Summary stage에서 tokenizer 접근이 필요하므로 PEFT wrapping 경로까지 설정
    model.tokenizer = tokenizer
    if hasattr(model, 'base_model'):
        model.base_model.tokenizer = tokenizer
        if hasattr(model.base_model, 'model'):
            model.base_model.model.tokenizer = tokenizer
    print(f"✓ Model loaded (context length: {context_len})")
    
    # 이미지 로드 및 전처리
    print(f"\n[2] Loading image: {image_file}")
    image = load_image(image_file)
    image_size = image.size
    print(f"✓ Image loaded: {image_size}")
    
    # 이미지 전처리
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 대화 템플릿 준비
    print(f"\n[3] Preparing conversation template")
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    # 이미지 토큰 포함한 프롬프트 생성
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    print(f"Prompt: {prompt}")
    
    # 토큰화
    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    print(f"Input tokens: {input_ids.shape}")
    
    # 생성
    print(f"\n[4] Generating response...")
    print(f"Using: {'Two-stage (Summary Tokens)' if use_summary_tokens else 'Standard (Full Image)'}")
    
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_summary_tokens=use_summary_tokens,  # 핵심 파라미터
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            use_cache=True,
        )
    
    # 디코딩
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    print(f"\n[5] Response:")
    print("=" * 70)
    print(outputs)
    print("=" * 70)
    
    return outputs


def compare_inference_modes(model_path, image_file, prompt, model_base=None):
    """
    Standard inference와 Two-stage inference 비교
    """
    print("\n" + "=" * 70)
    print("COMPARING STANDARD vs TWO-STAGE INFERENCE")
    print("=" * 70)
    
    # Standard inference
    print("\n" + "▶" * 35)
    response_standard = test_inference_basic(
        model_path, image_file, prompt, use_summary_tokens=False, model_base=model_base
    )
    
    # Two-stage inference
    print("\n" + "▶" * 35)
    response_twostage = test_inference_basic(
        model_path, image_file, prompt, use_summary_tokens=True, model_base=model_base
    )
    
    # 결과 비교
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print("\n[Standard Inference]")
    print(response_standard)
    print("\n[Two-stage Inference]")
    print(response_twostage)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (required for LoRA checkpoint inference)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image file or URL")
    parser.add_argument("--prompt", type=str, default="What is in this image?",
                        help="Question prompt")
    parser.add_argument("--use-summary-tokens", action="store_true",
                        help="Use two-stage inference with summary tokens")
    parser.add_argument("--compare", action="store_true",
                        help="Compare both inference modes")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_inference_modes(args.model_path, args.image, args.prompt, args.model_base)
    else:
        test_inference_basic(
            args.model_path, 
            args.image, 
            args.prompt,
            args.use_summary_tokens,
            args.model_base
        )
    
    print("\n✓ Inference test completed successfully!")
