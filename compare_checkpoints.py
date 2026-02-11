#!/usr/bin/env python3
"""
체크포인트 가중치 변화 분석 스크립트
최신 3개 체크포인트의 default adapter와 summary_utilizer adapter 가중치를 비교
"""

import os
import numpy as np
from safetensors import safe_open
from collections import OrderedDict

def load_checkpoint_weights(checkpoint_dir, adapter_name="default"):
    """체크포인트에서 가중치 로드"""
    if adapter_name == "default":
        weight_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    else:
        weight_path = os.path.join(checkpoint_dir, adapter_name, "adapter_model.safetensors")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    # Load weights from safetensors
    weights = {}
    with safe_open(weight_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    return weights

def compute_weight_stats(weights):
    """가중치의 통계 정보 계산"""
    all_values = []
    for name, param in weights.items():
        all_values.append(param.flatten())

    all_values = np.concatenate(all_values)

    return {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
        "abs_mean": float(np.mean(np.abs(all_values))),
        "total_params": len(all_values)
    }

def compare_weights(weights1, weights2, checkpoint1_name, checkpoint2_name):
    """두 체크포인트 간 가중치 차이 계산"""
    print(f"\n{'='*80}")
    print(f"Comparing: {checkpoint1_name} vs {checkpoint2_name}")
    print(f"{'='*80}")

    # 파라미터 이름 확인
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    if keys1 != keys2:
        print(f"⚠️  Parameter names mismatch!")
        print(f"   Only in {checkpoint1_name}: {keys1 - keys2}")
        print(f"   Only in {checkpoint2_name}: {keys2 - keys1}")
        common_keys = keys1 & keys2
    else:
        common_keys = keys1
        print(f"✅ All parameter names match ({len(common_keys)} parameters)")

    # 각 파라미터의 변화량 계산
    changes = OrderedDict()
    total_change = 0
    total_params = 0
    max_change = 0
    max_change_param = None

    for name in sorted(common_keys):
        param1 = weights1[name]
        param2 = weights2[name]

        if param1.shape != param2.shape:
            print(f"⚠️  Shape mismatch for {name}: {param1.shape} vs {param2.shape}")
            continue

        # 절대 차이
        diff = np.abs(param2 - param1)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        # 상대 차이 (평균 값 대비)
        mean_abs_value = float(np.mean(np.abs(param1)))
        relative_change = (mean_diff / (mean_abs_value + 1e-10)) * 100

        changes[name] = {
            "mean_abs_diff": mean_diff,
            "max_abs_diff": max_diff,
            "relative_change_%": relative_change,
            "param_count": param1.size
        }

        total_change += mean_diff * param1.size
        total_params += param1.size

        if max_diff > max_change:
            max_change = max_diff
            max_change_param = name

    # 전체 통계
    avg_change = total_change / total_params if total_params > 0 else 0

    print(f"\n📊 Overall Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Average absolute change: {avg_change:.6e}")
    print(f"   Maximum absolute change: {max_change:.6e}")
    print(f"   Max change parameter: {max_change_param}")

    # 변화가 큰 상위 10개 파라미터
    print(f"\n🔝 Top 10 parameters with largest mean absolute changes:")
    sorted_changes = sorted(changes.items(), key=lambda x: x[1]["mean_abs_diff"], reverse=True)

    for i, (name, stats) in enumerate(sorted_changes[:10], 1):
        print(f"   {i:2d}. {name}")
        print(f"       Mean abs diff: {stats['mean_abs_diff']:.6e} | "
              f"Max abs diff: {stats['max_abs_diff']:.6e} | "
              f"Relative: {stats['relative_change_%']:.2f}%")

    # 변화가 거의 없는 파라미터 확인
    no_change_params = [name for name, stats in changes.items() if stats["mean_abs_diff"] < 1e-10]
    if no_change_params:
        print(f"\n⚠️  Parameters with no change (< 1e-10): {len(no_change_params)}")
        for name in no_change_params[:5]:
            print(f"     - {name}")
        if len(no_change_params) > 5:
            print(f"     ... and {len(no_change_params) - 5} more")
    else:
        print(f"\n✅ All parameters have changed!")

    return avg_change, max_change

def main():
    checkpoint_base = "/home/deokhyeon/Documents/LLaVA/checkpoints/llava-v1.5-7b-token32-batch128"
    checkpoints = ["checkpoint-980", "checkpoint-990", "checkpoint-1000"]

    print("="*80)
    print("Dual-LoRA Checkpoint Weight Change Analysis")
    print("="*80)
    print(f"Base directory: {checkpoint_base}")
    print(f"Checkpoints: {', '.join(checkpoints)}")

    # 각 어댑터 분석
    for adapter_name in ["default", "summary_utilizer"]:
        print(f"\n\n{'#'*80}")
        print(f"# Analyzing: {adapter_name.upper()} Adapter")
        print(f"{'#'*80}")

        # 가중치 로드
        weights = {}
        for ckpt in checkpoints:
            ckpt_path = os.path.join(checkpoint_base, ckpt)
            try:
                weights[ckpt] = load_checkpoint_weights(ckpt_path, adapter_name)
                print(f"✅ Loaded {ckpt}/{adapter_name}")
            except Exception as e:
                print(f"❌ Failed to load {ckpt}/{adapter_name}: {e}")
                return

        # 각 체크포인트의 통계
        print(f"\n{'='*80}")
        print(f"Weight Statistics for Each Checkpoint")
        print(f"{'='*80}")

        for ckpt in checkpoints:
            stats = compute_weight_stats(weights[ckpt])
            print(f"\n{ckpt}:")
            print(f"   Mean: {stats['mean']:.6e} | Std: {stats['std']:.6e}")
            print(f"   Min: {stats['min']:.6e} | Max: {stats['max']:.6e}")
            print(f"   Abs Mean: {stats['abs_mean']:.6e}")
            print(f"   Total params: {stats['total_params']:,}")

        # 연속된 체크포인트 간 비교
        for i in range(len(checkpoints) - 1):
            compare_weights(
                weights[checkpoints[i]],
                weights[checkpoints[i+1]],
                checkpoints[i],
                checkpoints[i+1]
            )

        # 첫 번째와 마지막 비교 (전체 변화량)
        if len(checkpoints) > 2:
            print(f"\n\n{'='*80}")
            print(f"OVERALL CHANGE: {checkpoints[0]} → {checkpoints[-1]}")
            print(f"{'='*80}")
            compare_weights(
                weights[checkpoints[0]],
                weights[checkpoints[-1]],
                checkpoints[0],
                checkpoints[-1]
            )

    print(f"\n\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
