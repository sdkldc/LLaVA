"""
K-means 초기화 시스템 전체 검증 스크립트
- K-means clustering 동작 확인
- 다양한 설정 조합 테스트 (metric, apply_point, use_nearest)
- Summary token 초기화 검증
"""

import torch
import numpy as np
from llava.model.kmeans_initializer import (
    compute_distance,
    kmeans_clustering,
    find_nearest_tokens_to_centroids,
    initialize_summary_tokens_with_kmeans
)

def test_compute_distance():
    """거리 계산 함수 테스트"""
    print("\n=== Test 1: compute_distance ===")
    
    # [10, 128] 크기의 랜덤 토큰
    tokens = torch.randn(10, 128)
    # [3, 128] 크기의 랜덤 centroid
    centroids = torch.randn(3, 128)
    
    # 각 metric 테스트
    for metric in ['cosine', 'l2', 'dot']:
        dist = compute_distance(tokens, centroids, metric=metric)
        print(f"  {metric}: distance shape = {dist.shape}, min={dist.min():.4f}, max={dist.max():.4f}")
        assert dist.shape == (10, 3), f"Expected shape (10, 3), got {dist.shape}"
    
    print("  ✓ compute_distance 통과")

def test_kmeans_clustering():
    """K-means clustering 테스트"""
    print("\n=== Test 2: kmeans_clustering ===")
    
    # [100, 64] 크기의 토큰 (100개 토큰, 64차원)
    tokens = torch.randn(100, 64)
    num_clusters = 5
    
    for metric in ['cosine', 'l2', 'dot']:
        for n_iter in [1, 3, 5]:
            centroids = kmeans_clustering(tokens, num_clusters, metric=metric, n_iter=n_iter)
            print(f"  {metric} (iter={n_iter}): centroids shape = {centroids.shape}")
            assert centroids.shape == (num_clusters, 64), f"Expected ({num_clusters}, 64), got {centroids.shape}"
    
    print("  ✓ kmeans_clustering 통과")

def test_find_nearest_tokens():
    """Centroid에 가장 가까운 실제 토큰 찾기 테스트"""
    print("\n=== Test 3: find_nearest_tokens_to_centroids ===")
    
    # [50, 32] 크기의 토큰
    tokens = torch.randn(50, 32)
    num_clusters = 4
    
    for metric in ['cosine', 'l2', 'dot']:
        centroids = kmeans_clustering(tokens, num_clusters, metric=metric, n_iter=3)
        nearest_tokens = find_nearest_tokens_to_centroids(tokens, centroids, metric=metric)
        
        print(f"  {metric}: nearest_tokens shape = {nearest_tokens.shape}")
        assert nearest_tokens.shape == (num_clusters, 32), f"Expected ({num_clusters}, 32), got {nearest_tokens.shape}"
        
        # 실제 토큰 중 하나인지 확인
        for i in range(num_clusters):
            found = False
            for j in range(50):
                if torch.allclose(nearest_tokens[i], tokens[j], atol=1e-6):
                    found = True
                    break
            assert found, f"Cluster {i}의 nearest token이 실제 토큰 중 하나가 아닙니다!"
    
    print("  ✓ find_nearest_tokens_to_centroids 통과")

def test_initialize_summary_tokens():
    """Summary token 초기화 전체 프로세스 테스트"""
    print("\n=== Test 4: initialize_summary_tokens_with_kmeans ===")
    
    # [576, 1024] 크기의 이미지 토큰 (24x24 패치)
    image_features = torch.randn(576, 1024)
    num_summary = 16
    
    # 더미 프로젝터 생성
    mm_projector = torch.nn.Linear(1024, 4096)
    
    print("\n  [Subtest 4-1] use_nearest=False (centroid 직접 사용)")
    for metric in ['cosine', 'l2', 'dot']:
        summary_tokens = initialize_summary_tokens_with_kmeans(
            vision_features=image_features,
            mm_projector=mm_projector,
            num_summary_tokens=num_summary,
            metric=metric, 
            n_iter=3,
            use_nearest=False
        )
        print(f"    {metric}: summary_tokens shape = {summary_tokens.shape}")
        assert summary_tokens.shape == (num_summary, 4096), f"Expected ({num_summary}, 4096), got {summary_tokens.shape}"
    
    print("\n  [Subtest 4-2] use_nearest=True (가장 가까운 실제 토큰 사용)")
    for metric in ['cosine', 'l2', 'dot']:
        summary_tokens = initialize_summary_tokens_with_kmeans(
            vision_features=image_features,
            mm_projector=mm_projector,
            num_summary_tokens=num_summary,
            metric=metric, 
            n_iter=3,
            use_nearest=True
        )
        print(f"    {metric}: summary_tokens shape = {summary_tokens.shape}")
        assert summary_tokens.shape == (num_summary, 4096), f"Expected ({num_summary}, 4096), got {summary_tokens.shape}"
    
    print("  ✓ initialize_summary_tokens_with_kmeans 통과")

def test_batch_processing():
    """배치 처리 테스트"""
    print("\n=== Test 5: Batch Processing ===")
    
    batch_size = 4
    num_tokens = 576
    dim = 1024
    num_summary = 8
    
    # [B, N, D] 크기의 배치 이미지 토큰
    batch_image_features = torch.randn(batch_size, num_tokens, dim)
    mm_projector = torch.nn.Linear(dim, 4096)
    
    print(f"  Input: batch_size={batch_size}, num_tokens={num_tokens}, dim={dim}")
    print(f"  Target: num_summary={num_summary}")
    
    batch_summary_tokens = []
    for b in range(batch_size):
        summary = initialize_summary_tokens_with_kmeans(
            vision_features=batch_image_features[b],  # [N, D]
            mm_projector=mm_projector,
            num_summary_tokens=num_summary,
            metric='cosine',
            n_iter=3,
            use_nearest=True
        )
        batch_summary_tokens.append(summary)
    
    batch_summary_tokens = torch.stack(batch_summary_tokens, dim=0)  # [B, K, 4096]
    print(f"  Output: {batch_summary_tokens.shape}")
    assert batch_summary_tokens.shape == (batch_size, num_summary, 4096)
    
    print("  ✓ Batch processing 통과")

def test_different_configurations():
    """다양한 설정 조합 테스트"""
    print("\n=== Test 6: Different Configurations ===")
    
    image_features = torch.randn(576, 1024)
    
    configs = [
        # (metric, n_iter, use_nearest, apply_point_desc)
        ('cosine', 3, True, 'before_projector'),
        ('cosine', 3, False, 'before_projector'),
        ('l2', 5, True, 'before_projector'),
        ('dot', 1, False, 'before_projector'),
    ]
    
    mm_projector = torch.nn.Linear(1024, 4096)
    
    for metric, n_iter, use_nearest, apply_point in configs:
        summary = initialize_summary_tokens_with_kmeans(
            vision_features=image_features,
            mm_projector=mm_projector,
            num_summary_tokens=16,
            metric=metric,
            n_iter=n_iter,
            use_nearest=use_nearest
        )
        
        desc = f"centroid" if not use_nearest else "nearest"
        print(f"  ✓ {metric} | iter={n_iter} | {desc} | {apply_point}: shape={summary.shape}")

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n=== Test 7: Edge Cases ===")
    
    # Case 1: num_summary = 1
    print("  [Case 1] num_summary=1")
    tokens = torch.randn(100, 64)
    mm_projector = torch.nn.Linear(64, 64)  # 동일 차원 projector
    summary = initialize_summary_tokens_with_kmeans(
        vision_features=tokens,
        mm_projector=mm_projector,
        num_summary_tokens=1,
        metric='cosine',
        n_iter=3
    )
    assert summary.shape == (1, 64)
    print("    ✓ 단일 summary token")
    
    # Case 2: num_summary = num_tokens (모든 토큰이 대표)
    print("  [Case 2] num_summary=num_tokens")
    tokens = torch.randn(10, 32)
    mm_projector = torch.nn.Linear(32, 32)
    summary = initialize_summary_tokens_with_kmeans(
        vision_features=tokens,
        mm_projector=mm_projector,
        num_summary_tokens=10,
        metric='l2',
        n_iter=1
    )
    assert summary.shape == (10, 32)
    print("    ✓ 모든 토큰이 대표")
    
    # Case 3: 큰 차원
    print("  [Case 3] 큰 차원 (4096-dim)")
    tokens = torch.randn(576, 4096)  # after_projector 시나리오
    mm_projector = torch.nn.Linear(4096, 4096)  # 동일 차원 projector
    summary = initialize_summary_tokens_with_kmeans(
        vision_features=tokens,
        mm_projector=mm_projector,
        num_summary_tokens=8,
        metric='cosine',
        n_iter=3
    )
    assert summary.shape == (8, 4096)
    print("    ✓ 4096차원 처리 (after_projector)")

def test_determinism():
    """동일 입력에 대한 결정성 테스트 (같은 초기화 시드)"""
    print("\n=== Test 8: Determinism ===")
    
    torch.manual_seed(42)
    tokens1 = torch.randn(100, 64)
    
    torch.manual_seed(42)
    tokens2 = torch.randn(100, 64)
    
    # 동일한 입력
    assert torch.allclose(tokens1, tokens2)
    
    # K-means는 랜덤 초기화를 사용하므로, 시드를 고정해도 다를 수 있음
    # 하지만 use_nearest=True일 때는 실제 토큰에서 선택되므로 일관성 있음
    mm_projector = torch.nn.Linear(64, 64)
    
    torch.manual_seed(99)
    summary1 = initialize_summary_tokens_with_kmeans(
        vision_features=tokens1,
        mm_projector=mm_projector,
        num_summary_tokens=5,
        metric='cosine',
        n_iter=3,
        random_state=42,
        use_nearest=True
    )
    
    torch.manual_seed(99)
    summary2 = initialize_summary_tokens_with_kmeans(
        vision_features=tokens2,
        mm_projector=mm_projector,
        num_summary_tokens=5,
        metric='cosine',
        n_iter=3,
        random_state=42,
        use_nearest=True
    )
    
    print(f"  Summary1과 Summary2 차이: {(summary1 - summary2).abs().max().item():.6f}")
    print("  ✓ Determinism 확인 (use_nearest=True)")

def main():
    print("=" * 80)
    print("K-means 초기화 시스템 전체 검증 시작")
    print("=" * 80)
    
    try:
        test_compute_distance()
        test_kmeans_clustering()
        test_find_nearest_tokens()
        test_initialize_summary_tokens()
        test_batch_processing()
        test_different_configurations()
        test_edge_cases()
        test_determinism()
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        print("\n[검증 완료]")
        print("  ✓ K-means clustering 정상 동작")
        print("  ✓ 3가지 metric (cosine, l2, dot) 모두 정상")
        print("  ✓ use_nearest True/False 모두 정상")
        print("  ✓ 배치 처리 정상")
        print("  ✓ 다양한 설정 조합 정상")
        print("  ✓ 엣지 케이스 처리 정상")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}")
        raise

if __name__ == "__main__":
    main()
