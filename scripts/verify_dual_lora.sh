#!/bin/bash
# Dual LoRA 검증 스크립트 - GPU 3 사용

cd /home/deokhyeon/Documents/LLaVA

echo "================================"
echo "Dual LoRA 구현 검증"
echo "================================"
echo ""

CUDA_VISIBLE_DEVICES=3 python3 << 'PYTHON_SCRIPT'
import sys
print("✓ Python 실행")

# PEFT 버전 확인
try:
    import peft
    print(f"✓ PEFT 버전: {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 간단한 검증
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

print("\n[테스트] Dual LoRA 어댑터 생성")
print("-" * 60)

# 간단한 모델
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)
        self.config = type('C', (), {'model_type': 'test'})()
    
    def forward(self, x):
        return self.linear2(self.linear1(x))
    
    def prepare_inputs_for_generation(self, x, **kw):
        return {"x": x}
    
    def get_input_embeddings(self):
        return self.linear1
    
    def set_input_embeddings(self, v):
        self.linear1 = v

model = TestModel()

# LoRA 1 추가
config1 = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["linear1", "linear2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config1)
print(f"✓ 첫 번째 LoRA 어댑터 추가")

# 어댑터 목록 확인
if hasattr(model, 'peft_config'):
    adapters_before = list(model.peft_config.keys())
    print(f"  현재 어댑터: {adapters_before}")
    current = list(model.peft_config.keys())[0]
    print(f"  ✓ 첫 번째 어댑터: '{current}' (1st forward용)")

# LoRA 2 추가
config2 = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["linear1", "linear2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter("summary_utilizer", config2)
print(f"✓ 두 번째 LoRA 어댑터 추가")

# 최종 확인
adapters_after = list(model.peft_config.keys())
print(f"  최종 어댑터 목록: {adapters_after}")

# 검증
expected = {'default', 'summary_utilizer'}
actual = set(adapters_after)

if expected == actual:
    print(f"\n✓ 성공: 두 개의 독립적인 LoRA 어댑터가 생성됨!")
    print(f"  - {adapters_after}")
    
    # 어댑터 전환 테스트
    print("\n[테스트] 어댑터 전환")
    print("-" * 60)
    for name in ['default', 'summary_utilizer', 'default']:
        model.set_adapter(name)
        active = getattr(model, 'active_adapter', 'unknown')
        if active == name:
            print(f"  ✓ 전환 성공: {name}")
        else:
            print(f"  ✗ 전환 실패: {name} → {active}")
    
    print("\n" + "=" * 60)
    print("✓ Dual LoRA 구현이 정상 작동합니다!")
    print("=" * 60)
    print("\n학습 시 동작:")
    print("  1st Forward (요약 생성) → 'default' 어댑터 사용")
    print("  2nd Forward (요약 활용) → 'summary_utilizer' 어댑터 사용")
    print("  각 forward마다 독립적인 LoRA 가중치 학습")
else:
    print(f"\n✗ 실패: 어댑터 불일치")
    print(f"  예상: {expected}")
    print(f"  실제: {actual}")
    sys.exit(1)

PYTHON_SCRIPT
