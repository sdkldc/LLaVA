#!/usr/bin/env python3
"""
Gradient 정보 확인 스크립트
어떤 파라미터가 requires_grad=True/False인지 확인
"""
import torch
from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

def check_gradients():
    print("="*80)
    print("Gradient 상태 확인")
    print("="*80)
    
    # 모델 로드
    model_path = './checkpoints/llava-v1.5-7b'
    print(f"\n모델 로드 중: {model_path}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    # 1. 기본 상태 확인
    print("\n" + "="*80)
    print("1. 모델 로드 직후 상태")
    print("="*80)
    check_model_grads(model, "모델 로드 직후")
    
    # 2. Backbone freeze
    print("\n" + "="*80)
    print("2. Backbone freeze 후")
    print("="*80)
    model.model.requires_grad_(False)
    check_model_grads(model, "Backbone freeze 후")
    
    # 3. LoRA 추가
    print("\n" + "="*80)
    print("3. LoRA 추가 후")
    print("="*80)
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    check_model_grads(model, "LoRA 추가 후")
    
    # 4. MM Projector 학습 설정
    print("\n" + "="*80)
    print("4. MM Projector requires_grad=True 설정 후")
    print("="*80)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True
    check_model_grads(model, "MM Projector 활성화 후")
    
    # 5. Vision Tower 확인
    print("\n" + "="*80)
    print("5. Vision Tower 상태")
    print("="*80)
    vision_tower = model.get_model().get_vision_tower()
    total = 0
    trainable = 0
    for name, param in vision_tower.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Vision Tower:")
    print(f"  - 총 파라미터: {total:,}")
    print(f"  - 학습 가능: {trainable:,}")
    print(f"  - Frozen: {'✓' if trainable == 0 else '✗'}")
    
    # 6. 상세 분석
    print("\n" + "="*80)
    print("6. 상세 분석: requires_grad=True인 파라미터들")
    print("="*80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name:60s} | shape: {str(param.shape):20s} | dtype: {param.dtype}")


def check_model_grads(model, stage_name):
    """모델의 각 컴포넌트별 gradient 상태 확인"""
    total_params = 0
    trainable_params = 0
    
    # 컴포넌트별로 체크
    components = {
        'Vision Tower': [],
        'MM Projector': [],
        'LLM Backbone': [],
        'LoRA': [],
        'Summary Tokens': [],
        'Others': []
    }
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
        # 컴포넌트 분류
        if 'vision_tower' in name:
            components['Vision Tower'].append((name, param))
        elif 'mm_projector' in name:
            components['MM Projector'].append((name, param))
        elif 'lora' in name.lower():
            components['LoRA'].append((name, param))
        elif 'summary_tokens' in name:
            components['Summary Tokens'].append((name, param))
        elif any(x in name for x in ['model.layers', 'model.embed_tokens', 'model.norm', 'lm_head']):
            components['LLM Backbone'].append((name, param))
        else:
            components['Others'].append((name, param))
    
    print(f"\n{stage_name}:")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"\n  컴포넌트별:")
    
    for comp_name, params in components.items():
        if not params:
            continue
        comp_total = sum(p.numel() for _, p in params)
        comp_trainable = sum(p.numel() for _, p in params if p.requires_grad)
        status = "학습" if comp_trainable > 0 else "Frozen"
        print(f"    {comp_name:20s}: {comp_trainable:10,} / {comp_total:10,} ({status})")


if __name__ == '__main__':
    check_gradients()
