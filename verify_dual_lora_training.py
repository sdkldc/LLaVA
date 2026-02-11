#!/usr/bin/env python3
"""
Dual-LoRA 학습 검증: 두 adapter가 모두 학습되는지 확인
"""

import torch
import torch.nn as nn

class DummyLoRALayer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.weight = nn.Parameter(torch.randn(10, 10) * 0.01)

    def forward(self, x):
        return torch.matmul(x, self.weight)

# Simulated dual-LoRA model
class DualLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict({
            'summary_utilizer': DummyLoRALayer('summary_utilizer'),
            'default': DummyLoRALayer('default')
        })
        self.active_adapter = 'summary_utilizer'

    def set_adapter(self, name):
        self.active_adapter = name

    def forward(self, x):
        return self.adapters[self.active_adapter](x)

print("="*80)
print("Dual-LoRA Training Flow Verification")
print("="*80)

model = DualLoRAModel()
x = torch.randn(5, 10, requires_grad=True)

print("\n📊 Initial weights:")
print(f"  summary_utilizer: mean={model.adapters['summary_utilizer'].weight.mean().item():.6f}")
print(f"  default: mean={model.adapters['default'].weight.mean().item():.6f}")

# ========== Test Case 1: WITH 1st backward ==========
print("\n" + "="*80)
print("Test Case 1: WITH 1st backward (aux_loss)")
print("="*80)

model1 = DualLoRAModel()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)

# Save initial weights
init_summary = model1.adapters['summary_utilizer'].weight.clone()
init_default = model1.adapters['default'].weight.clone()

# 1st forward (summary_utilizer)
model1.set_adapter('summary_utilizer')
summary_hidden = model1(x)  # Shape: [5, 10]

# 1st backward with aux_loss
aux_loss = summary_hidden.pow(2).mean()
aux_loss.backward(retain_graph=True)  # ✅ retain_graph=True!

print(f"✅ 1st backward done (aux_loss={aux_loss.item():.4f})")

# Switch adapter
model1.set_adapter('default')

# 2nd forward (default)
output = model1(summary_hidden)  # Use summary as input
main_loss = output.pow(2).mean()

# 2nd backward
main_loss.backward()

print(f"✅ 2nd backward done (main_loss={main_loss.item():.4f})")

# Check gradients
print("\n📊 Gradients:")
summary_grad = model1.adapters['summary_utilizer'].weight.grad
default_grad = model1.adapters['default'].weight.grad

if summary_grad is not None:
    print(f"  summary_utilizer: grad_mean={summary_grad.mean().item():.6f}, grad_norm={summary_grad.norm().item():.6f}")
else:
    print(f"  summary_utilizer: ❌ NO GRADIENT!")

if default_grad is not None:
    print(f"  default: grad_mean={default_grad.mean().item():.6f}, grad_norm={default_grad.norm().item():.6f}")
else:
    print(f"  default: ❌ NO GRADIENT!")

# Update weights
optimizer1.step()

# Check weight changes
summary_change = (model1.adapters['summary_utilizer'].weight - init_summary).abs().mean().item()
default_change = (model1.adapters['default'].weight - init_default).abs().mean().item()

print("\n📊 Weight changes:")
print(f"  summary_utilizer: {summary_change:.6f}")
print(f"  default: {default_change:.6f}")

if summary_change > 1e-6 and default_change > 1e-6:
    print("\n✅ PASS: Both adapters updated!")
else:
    print("\n❌ FAIL: Some adapter not updated!")

# ========== Test Case 2: WITHOUT 1st backward ==========
print("\n" + "="*80)
print("Test Case 2: WITHOUT 1st backward (aux_loss = None)")
print("="*80)

model2 = DualLoRAModel()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

# Save initial weights
init_summary2 = model2.adapters['summary_utilizer'].weight.clone()
init_default2 = model2.adapters['default'].weight.clone()

# 1st forward (summary_utilizer)
model2.set_adapter('summary_utilizer')
summary_hidden2 = model2(x)

print(f"❌ 1st backward SKIPPED (aux_loss=None)")

# Switch adapter
model2.set_adapter('default')

# 2nd forward (default)
output2 = model2(summary_hidden2)
main_loss2 = output2.pow(2).mean()

# 2nd backward
main_loss2.backward()

print(f"✅ 2nd backward done (main_loss={main_loss2.item():.4f})")

# Check gradients
print("\n📊 Gradients:")
summary_grad2 = model2.adapters['summary_utilizer'].weight.grad
default_grad2 = model2.adapters['default'].weight.grad

if summary_grad2 is not None:
    print(f"  summary_utilizer: grad_mean={summary_grad2.mean().item():.6f}, grad_norm={summary_grad2.norm().item():.6f}")
else:
    print(f"  summary_utilizer: ❌ NO GRADIENT!")

if default_grad2 is not None:
    print(f"  default: grad_mean={default_grad2.mean().item():.6f}, grad_norm={default_grad2.norm().item():.6f}")
else:
    print(f"  default: ❌ NO GRADIENT!")

# Update weights
optimizer2.step()

# Check weight changes
summary_change2 = (model2.adapters['summary_utilizer'].weight - init_summary2).abs().mean().item()
default_change2 = (model2.adapters['default'].weight - init_default2).abs().mean().item()

print("\n📊 Weight changes:")
print(f"  summary_utilizer: {summary_change2:.6f}")
print(f"  default: {default_change2:.6f}")

if summary_change2 > 1e-6 and default_change2 > 1e-6:
    print("\n✅ PASS: Both adapters updated!")
else:
    print("\n❌ FAIL: Some adapter not updated!")
    if summary_change2 < 1e-6:
        print("   → summary_utilizer NOT UPDATED!")
    if default_change2 < 1e-6:
        print("   → default NOT UPDATED!")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Test 1 (WITH 1st backward): summary_utilizer updated = {summary_change > 1e-6}")
print(f"Test 2 (WITHOUT 1st backward): summary_utilizer updated = {summary_change2 > 1e-6}")
print("\n💡 Conclusion:")
if summary_change2 < 1e-6:
    print("   ❌ WITHOUT aux_loss, summary_utilizer adapter is NOT trained!")
    print("   → Must set --summary_aux_loss_weight > 0")
else:
    print("   ✅ End-to-end gradient works even without aux_loss")
