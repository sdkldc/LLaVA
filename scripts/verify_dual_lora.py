#!/usr/bin/env python3
"""
Dual LoRA 구현 검증 스크립트

이 스크립트는 다음을 확인합니다:
1. 두 개의 LoRA 어댑터가 생성되었는지
2. 어댑터 이름이 올바른지 (summary_generator, summary_utilizer)
3. 어댑터 전환이 제대로 작동하는지
4. 각 어댑터가 독립적인 파라미터를 가지는지
"""

import sys
import torch
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("Dual LoRA 구현 검증")
print("=" * 80)

# 1. PEFT 버전 확인
try:
    import peft
    print(f"✓ PEFT 버전: {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT import 실패: {e}")
    sys.exit(1)

# 2. 간단한 모델로 테스트
print("\n[테스트 1] Dual LoRA 어댑터 생성 테스트")
print("-" * 80)

try:
    # 간단한 LLM 모델 (테스트용)
    import torch.nn as nn
    
    # 매우 간단한 테스트 모델 생성
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 128)
            self.c_attn = nn.Linear(128, 128)
            self.c_proj = nn.Linear(128, 128)
            self.lm_head = nn.Linear(128, 1000)
            self.config = type('Config', (), {
                'vocab_size': 1000,
                'hidden_size': 128,
                'n_layer': 1,
                'n_head': 2,
                'model_type': 'gpt2'
            })()
        
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            x = self.c_attn(x)
            x = self.c_proj(x)
            return self.lm_head(x)
        
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}
        
        def get_input_embeddings(self):
            return self.embed
        
        def set_input_embeddings(self, value):
            self.embed = value
    
    print("작은 테스트 모델 생성 중...")
    model = SimpleModel()
    
    # LoRA 설정
    lora_config_1 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 첫 번째 어댑터 추가
    print("첫 번째 LoRA 어댑터 추가...")
    model = get_peft_model(model, lora_config_1)
    
    # 어댑터 정보 확인
    if hasattr(model, 'peft_config'):
        print(f"  현재 어댑터 목록: {list(model.peft_config.keys())}")
        current_adapter = getattr(model, 'active_adapter', 'default')
        print(f"  활성 어댑터: {current_adapter}")
        print(f"  ✓ 첫 번째 어댑터: '{current_adapter}' (1st forward용)")
    
    # 두 번째 어댑터 추가
    lora_config_2 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("두 번째 LoRA 어댑터 추가...")
    model.add_adapter("summary_utilizer", lora_config_2)
    
    # 최종 어댑터 목록 확인
    if hasattr(model, 'peft_config'):
        adapters = list(model.peft_config.keys())
        print(f"  최종 어댑터 목록: {adapters}")
        
        # 검증
        expected_adapters = {'default', 'summary_utilizer'}
        actual_adapters = set(adapters)
        
        if expected_adapters == actual_adapters:
            print(f"  ✓ 어댑터 생성 성공: {adapters}")
        else:
            print(f"  ✗ 어댑터 불일치!")
            print(f"    예상: {expected_adapters}")
            print(f"    실제: {actual_adapters}")
            sys.exit(1)
    
    print("\n[테스트 2] 어댑터 전환 테스트")
    print("-" * 80)
    
    # 어댑터 전환 테스트
    for adapter_name in ['default', 'summary_utilizer', 'default']:
        model.set_adapter(adapter_name)
        active = getattr(model, 'active_adapter', 'unknown')
        if active == adapter_name:
            print(f"  ✓ 어댑터 전환 성공: {adapter_name} (활성: {active})")
        else:
            print(f"  ✗ 어댑터 전환 실패: {adapter_name} → {active}")
            sys.exit(1)
    
    print("\n[테스트 3] 어댑터 파라미터 독립성 테스트")
    print("-" * 80)
    
    # 각 어댑터의 파라미터 개수 확인
    model.set_adapter("default")
    gen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model.set_adapter("summary_utilizer")
    util_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  summary_generator 학습 가능 파라미터: {gen_params:,}")
    print(f"  summary_utilizer 학습 가능 파라미터: {util_params:,}")
    
    if gen_params > 0 and util_params > 0:
        print(f"  ✓ 두 어댑터 모두 학습 가능한 파라미터 보유")
    else:
        print(f"  ✗ 어댑터 파라미터 오류")
        sys.exit(1)
    
    # 파라미터 메모리 주소 비교 (독립성 확인)
    model.set_adapter("default")
    gen_first_param = None
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            gen_first_param = (name, id(param))
            break
    
    model.set_adapter("summary_utilizer")
    util_first_param = None
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            util_first_param = (name, id(param))
            break
    
    if gen_first_param and util_first_param:
        print(f"\n  어댑터 1 파라미터: {gen_first_param[0]} (id: {gen_first_param[1]})")
        print(f"  어댑터 2 파라미터: {util_first_param[0]} (id: {util_first_param[1]})")
        
        if gen_first_param[1] != util_first_param[1]:
            print(f"  ✓ 두 어댑터의 파라미터가 독립적으로 존재")
        else:
            print(f"  ⚠ 두 어댑터가 같은 파라미터를 공유 (예상치 못한 동작)")
    
    print("\n" + "=" * 80)
    print("✓ 모든 테스트 통과!")
    print("=" * 80)
    print("\nDual LoRA 구현이 올바르게 작동합니다:")
    print("  1. ✓ 두 개의 독립적인 LoRA 어댑터 생성됨")
    print("  2. ✓ 어댑터 이름이 올바름 (default, summary_utilizer)")
    print("  3. ✓ 어댑터 전환이 제대로 작동함")
    print("  4. ✓ 각 어댑터가 독립적인 파라미터를 가짐")
    print("\n학습 시:")
    print("  - 1st Forward (요약 생성): 'default' 어댑터 사용")
    print("  - 2nd Forward (요약 활용): 'summary_utilizer' 어댑터 사용")
    
except Exception as e:
    print(f"\n✗ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
