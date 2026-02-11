#!/bin/bash

# GPU 1에서 모든 테스트 실행
export CUDA_VISIBLE_DEVICES=1

echo "========================================="
echo "Starting all tests on GPU 1"
echo "========================================="
echo ""

# 테스트 파일 리스트 (루트 디렉토리)
tests=(
    "test_attention_mask.py"
    "test_dynamic_kmeans.py"
    "test_inference_structure.py"
    "test_kmeans_apply_point.py"
    "test_kmeans_full.py"
    "test_kmeans_init.py"
    "test_kmeans_quick.py"
    "test_kmeans_simple.py"
    "test_summary_token_generation.py"
    "test_end_to_end.py"  # End-to-end 통합 테스트
)

# 각 테스트 실행
for test in "${tests[@]}"; do
    echo "========================================="
    echo "Running: $test"
    echo "========================================="
    
    python3 "$test"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $test PASSED"
    else
        echo "✗ $test FAILED (exit code: $exit_code)"
    fi
    
    echo ""
    echo ""
done

echo "========================================="
echo "All tests completed"
echo "========================================="
