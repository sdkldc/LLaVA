#!/usr/bin/env python3
"""
Summary Tokens 품질 비교 평가

Standard inference vs Two-stage inference의 출력 품질 비교
"""

import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

import sys
sys.path.append('/home/deokhyeon/Documents/LLaVA')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def load_image(image_path):
    """이미지 로드"""
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image


def run_inference(model, tokenizer, image_processor, image, prompt, use_summary_tokens=False):
    """단일 이미지에 대한 추론 실행"""
    image_size = image.size
    
    # 이미지 전처리
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 프롬프트 준비
    conv = conv_templates["llava_v1"].copy()
    full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # 토큰화
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # 생성
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_summary_tokens=use_summary_tokens,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            use_cache=True,
        )
    
    # 디코딩
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


def evaluate_dataset(model_path, dataset_file, output_file):
    """데이터셋 전체 평가"""
    print(f"Loading model from: {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map='auto'
    )
    print(f"✓ Model loaded")
    
    # 데이터셋 로드
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Dataset: {len(dataset)} samples")
    
    # 평가 실행
    results = []
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        image_path = item['image']
        prompt = item['prompt']
        
        try:
            # 이미지 로드
            image = load_image(image_path)
            
            # Standard inference
            output_standard = run_inference(
                model, tokenizer, image_processor, image, prompt, 
                use_summary_tokens=False
            )
            
            # Two-stage inference
            output_twostage = run_inference(
                model, tokenizer, image_processor, image, prompt, 
                use_summary_tokens=True
            )
            
            # 결과 저장
            result = {
                'id': idx,
                'image': image_path,
                'prompt': prompt,
                'output_standard': output_standard,
                'output_twostage': output_twostage,
                'ground_truth': item.get('answer', None)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue
    
    # 결과 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # 간단한 통계
    print(f"\nProcessed: {len(results)}/{len(dataset)} samples")
    
    return results


def analyze_results(results_file):
    """결과 분석"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("Quality Comparison Analysis")
    print("="*70)
    
    # 길이 비교
    len_standard = [len(r['output_standard']) for r in results]
    len_twostage = [len(r['output_twostage']) for r in results]
    
    print(f"\nOutput Length:")
    print(f"  Standard:  {sum(len_standard)/len(len_standard):.1f} chars (avg)")
    print(f"  Two-stage: {sum(len_twostage)/len(len_twostage):.1f} chars (avg)")
    
    # 샘플 출력
    print(f"\n{'='*70}")
    print("Sample Outputs (first 3)")
    print("="*70)
    
    for i, result in enumerate(results[:3]):
        print(f"\n[Sample {i+1}]")
        print(f"Prompt: {result['prompt']}")
        print(f"\nStandard:\n{result['output_standard'][:200]}...")
        print(f"\nTwo-stage:\n{result['output_twostage'][:200]}...")
        print("-"*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="JSON file with image paths and prompts")
    parser.add_argument("--output", type=str, default="quality_comparison_results.json")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results file")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_results(args.output)
    else:
        results = evaluate_dataset(args.model_path, args.dataset, args.output)
        analyze_results(args.output)
