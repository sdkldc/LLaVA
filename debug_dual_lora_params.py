"""
Dual LoRA 모델의 파라미터 이름 구조 확인
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./checkpoints/llava-v1.5-7b"
lora_path = "./checkpoints/llava-v1.5-7b-batch128-token32"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Loading default adapter...")
model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")

print("Loading summary_utilizer adapter...")
summary_utilizer_path = f"{lora_path}/summary_utilizer"
try:
    model.load_adapter(summary_utilizer_path, adapter_name="summary_utilizer")
    print("✓ summary_utilizer adapter loaded")
except Exception as e:
    print(f"✗ Failed to load summary_utilizer: {e}")

print("\nActive adapters:", model.active_adapters if hasattr(model, 'active_adapters') else 'N/A')
print("PEFT config keys:", list(model.peft_config.keys()))

# 파라미터 이름 확인
print("\n=== Checking parameter names ===")
default_params = []
utilizer_params = []
other_params = []

for name, param in model.named_parameters():
    if 'lora_' in name:
        # LoRA 파라미터인 경우
        if 'default' in name:
            default_params.append(name)
        elif 'summary_utilizer' in name:
            utilizer_params.append(name)
        else:
            other_params.append(name)

print(f"\nDefault adapter params: {len(default_params)}")
if default_params:
    print("  Sample:", default_params[0])

print(f"\nSummary_utilizer adapter params: {len(utilizer_params)}")
if utilizer_params:
    print("  Sample:", utilizer_params[0])

print(f"\nOther LoRA params: {len(other_params)}")
if other_params:
    print("  First 5 samples:")
    for p in other_params[:5]:
        print(f"    {p}")

# 어댑터 전환 테스트
print("\n=== Testing adapter switching ===")
model.set_adapter("default")
print("Active adapter: default")
for name, param in model.named_parameters():
    if 'lora_' in name and param.requires_grad:
        print(f"  Trainable: {name[:80]}...")
        break

model.set_adapter("summary_utilizer")
print("\nActive adapter: summary_utilizer")
for name, param in model.named_parameters():
    if 'lora_' in name and param.requires_grad:
        print(f"  Trainable: {name[:80]}...")
        break
