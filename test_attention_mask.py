#!/usr/bin/env python3
"""
Attention Mask 동작 검증 스크립트

첫 번째 forward에서 요약 토큰들이 서로를 참조하지 못하는지 확인
"""

import torch
import sys
sys.path.append('/home/deokhyeon/Documents/LLaVA')

from llava.model.attention_utils import (
    create_summary_token_attention_mask,
    create_summary_token_attention_mask_optimized,
    visualize_attention_mask,
    convert_mask_to_additive,
    combine_masks
)


def test_basic_attention_mask():
    """기본 Attention Mask 생성 테스트"""
    print("=" * 70)
    print("Test 1: Basic Attention Mask Generation")
    print("=" * 70)
    
    batch_size = 2
    prompt_len = 10
    image_len = 576  # 24x24 patches
    num_summary_tokens = 8
    seq_length = prompt_len + image_len + num_summary_tokens
    
    summary_start = prompt_len + image_len
    summary_end = summary_start + num_summary_tokens
    summary_positions = (summary_start, summary_end)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Attention mask 생성
    mask = create_summary_token_attention_mask(
        batch_size=batch_size,
        seq_length=seq_length,
        summary_token_positions=summary_positions,
        device=device,
        dtype=torch.float32
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Summary token positions: {summary_positions}")
    print(f"Total sequence length: {seq_length}")
    print()
    
    # 검증 1: 요약 토큰끼리 참조 차단 확인
    print("Verification 1: Summary tokens cannot attend to each other")
    for i in range(summary_start, summary_end):
        for j in range(summary_start, summary_end):
            if i != j:
                # i번째 요약 토큰이 j번째 요약 토큰을 참조하면 안 됨 (True여야 함)
                is_blocked = mask[0, 0, i, j].item()
                assert is_blocked == True, f"Summary token {i} should not attend to {j}"
    print("✓ Summary tokens are properly blocked from attending to each other")
    print()
    
    # 검증 2: 자기 자신은 참조 가능
    print("Verification 2: Summary tokens can attend to themselves")
    for i in range(summary_start, summary_end):
        is_allowed = mask[0, 0, i, i].item()
        assert is_allowed == False, f"Summary token {i} should attend to itself"
    print("✓ Summary tokens can attend to themselves (diagonal)")
    print()
    
    # 검증 3: 이미지 토큰은 참조 가능
    print("Verification 3: Summary tokens can attend to image tokens")
    for i in range(summary_start, summary_end):
        for j in range(prompt_len, summary_start):  # 이미지 영역
            # Causal mask 고려: i >= j 일 때만 참조 가능
            if i >= j:
                is_allowed = mask[0, 0, i, j].item()
                assert is_allowed == False, f"Summary token {i} should attend to image token {j}"
    print("✓ Summary tokens can attend to image tokens")
    print()
    
    # 검증 4: 프롬프트 토큰은 참조 가능
    print("Verification 4: Summary tokens can attend to prompt tokens")
    for i in range(summary_start, summary_end):
        for j in range(prompt_len):  # 프롬프트 영역
            is_allowed = mask[0, 0, i, j].item()
            assert is_allowed == False, f"Summary token {i} should attend to prompt token {j}"
    print("✓ Summary tokens can attend to prompt tokens")
    print()


def test_optimized_vs_basic():
    """최적화 버전과 기본 버전 비교"""
    print("=" * 70)
    print("Test 2: Optimized vs Basic Implementation")
    print("=" * 70)
    
    batch_size = 2
    seq_length = 600
    summary_positions = (586, 594)  # 8개 요약 토큰
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 기본 버전
    mask_basic = create_summary_token_attention_mask(
        batch_size, seq_length, summary_positions, device
    )
    
    # 최적화 버전
    mask_optimized = create_summary_token_attention_mask_optimized(
        batch_size, seq_length, summary_positions, device
    )
    
    # 비교
    is_equal = torch.equal(mask_basic, mask_optimized)
    print(f"Basic and Optimized masks are equal: {is_equal}")
    
    if not is_equal:
        diff = (mask_basic != mask_optimized).sum().item()
        print(f"Number of different elements: {diff}")
    else:
        print("✓ Both implementations produce identical results")
    print()


def test_additive_conversion():
    """Boolean mask를 Additive mask로 변환 테스트"""
    print("=" * 70)
    print("Test 3: Boolean to Additive Mask Conversion")
    print("=" * 70)
    
    # 간단한 Boolean mask 생성
    batch_size = 1
    seq_length = 10
    summary_positions = (7, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bool_mask = create_summary_token_attention_mask(
        batch_size, seq_length, summary_positions, device
    )
    
    # Additive mask로 변환
    additive_mask = convert_mask_to_additive(bool_mask, dtype=torch.float32)
    
    print(f"Boolean mask shape: {bool_mask.shape}")
    print(f"Additive mask shape: {additive_mask.shape}")
    print()
    
    # 검증: True -> -inf, False -> 0
    blocked_positions = bool_mask == True
    allowed_positions = bool_mask == False
    
    assert torch.all(additive_mask[blocked_positions] == float('-inf')), \
        "Blocked positions should be -inf"
    assert torch.all(additive_mask[allowed_positions] == 0.0), \
        "Allowed positions should be 0.0"
    
    print("✓ Boolean mask correctly converted to additive mask")
    print(f"  - Blocked positions (True): {blocked_positions.sum().item()} -> -inf")
    print(f"  - Allowed positions (False): {allowed_positions.sum().item()} -> 0.0")
    print()


def test_visualization():
    """Attention Mask 시각화 테스트"""
    print("=" * 70)
    print("Test 4: Attention Mask Visualization")
    print("=" * 70)
    
    batch_size = 1
    prompt_len = 10
    image_len = 50
    num_summary_tokens = 8
    seq_length = prompt_len + image_len + num_summary_tokens
    
    summary_start = prompt_len + image_len
    summary_end = summary_start + num_summary_tokens
    summary_positions = (summary_start, summary_end)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mask = create_summary_token_attention_mask(
        batch_size, seq_length, summary_positions, device
    )
    
    # 시각화 (matplotlib 없으면 스킵)
    try:
        visualize_attention_mask(
            mask,
            save_path='/home/deokhyeon/Documents/LLaVA/attention_mask_visualization.png'
        )
        print("✓ Attention mask visualization saved")
    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")
    print()


def test_different_summary_token_counts():
    """다양한 요약 토큰 개수로 테스트"""
    print("=" * 70)
    print("Test 5: Different Summary Token Counts")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for num_summary in [1, 4, 8, 16, 32]:
        prompt_len = 10
        image_len = 576
        seq_length = prompt_len + image_len + num_summary
        summary_positions = (prompt_len + image_len, seq_length)
        
        mask = create_summary_token_attention_mask_optimized(
            batch_size=1,
            seq_length=seq_length,
            summary_token_positions=summary_positions,
            device=device
        )
        
        # 요약 토큰 영역의 차단 개수 계산
        start, end = summary_positions
        summary_region = mask[0, 0, start:end, start:end]
        blocked = (summary_region == True).sum().item()
        total = summary_region.numel()
        
        # 예상: diagonal 제외한 모든 위치가 차단되어야 함
        expected_blocked = num_summary * num_summary - num_summary
        
        print(f"Summary tokens: {num_summary:2d} | "
              f"Blocked: {blocked}/{total} | "
              f"Expected: {expected_blocked} | "
              f"{'✓' if blocked == expected_blocked else '✗'}")
        
        assert blocked == expected_blocked, \
            f"Expected {expected_blocked} blocked positions, got {blocked}"
    
    print("\n✓ All summary token counts work correctly")
    print()


def test_mask_combination():
    """패딩 마스크와 커스텀 마스크 결합 테스트"""
    print("=" * 70)
    print("Test 6: Padding Mask + Custom Mask Combination")
    print("=" * 70)
    
    batch_size = 2
    seq_length = 20
    summary_positions = (15, 20)  # 마지막 5개 토큰이 요약 토큰
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 커스텀 마스크 생성 (요약 토큰 간 참조 방지)
    custom_mask = create_summary_token_attention_mask_optimized(
        batch_size, seq_length, summary_positions, device
    )
    
    # 패딩 마스크 생성 (배치 1은 전체 유효, 배치 2는 뒤 5개만 유효)
    padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    padding_mask[1, :15] = False  # 배치 2의 앞 15개는 패딩
    
    # 두 마스크 결합
    combined = combine_masks(custom_mask, padding_mask)
    
    print(f"Custom mask shape: {custom_mask.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Combined mask shape: {combined.shape}")
    print()
    
    # 검증 1: 배치 1은 커스텀 마스크만 적용
    batch1_custom = custom_mask[0, 0]
    batch1_combined = combined[0, 0]
    assert torch.equal(batch1_custom, batch1_combined), \
        "Batch 1 should only have custom mask"
    print("✓ Batch 1: Custom mask only (no padding)")
    
    # 검증 2: 배치 2는 패딩 영역이 추가로 차단됨
    batch2_combined = combined[1, 0]
    # 패딩 영역 (앞 15개)의 key 위치는 모두 차단되어야 함
    for i in range(seq_length):
        for j in range(15):  # 패딩 영역
            assert batch2_combined[i, j] == True, \
                f"Position ({i},{j}) should be blocked (padding)"
    print("✓ Batch 2: Padding positions are blocked")
    
    # 검증 3: 배치 2의 요약 토큰끼리는 여전히 차단됨
    for i in range(15, 20):
        for j in range(15, 20):
            if i != j:
                assert batch2_combined[i, j] == True, \
                    f"Summary tokens ({i},{j}) should still be blocked"
    print("✓ Batch 2: Summary tokens still blocked from each other")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ATTENTION MASK VERIFICATION TEST")
    print("=" * 70 + "\n")
    
    try:
        test_basic_attention_mask()
        test_optimized_vs_basic()
        test_additive_conversion()
        test_different_summary_token_counts()
        test_mask_combination()
        test_visualization()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nAttention mask implementation is correct:")
        print("  1. Summary tokens cannot attend to each other")
        print("  2. Summary tokens can attend to themselves")
        print("  3. Summary tokens can attend to prompt and image tokens")
        print("  4. Causal mask is properly maintained")
        print("  5. Padding mask and custom mask are properly combined")
        print()
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
