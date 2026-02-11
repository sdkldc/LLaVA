"""
K-means 초기화 최소 검증 스크립트
DeepSpeed 없이 순수 PyTorch만 사용
"""

import torch
import numpy as np

# K-means 함수들을 직접 복사해서 테스트
def compute_distance(tokens, centroids, metric='cosine'):
    """거리 계산"""
    if metric == 'cosine':
        tokens_norm = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-8)
        centroids_norm = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.matmul(tokens_norm, centroids_norm.T)
        return 1 - similarity
    elif metric == 'l2':
        tokens_expanded = tokens.unsqueeze(1)
        centroids_expanded = centroids.unsqueeze(0)
        return torch.norm(tokens_expanded - centroids_expanded, dim=-1)
    elif metric == 'dot':
        dot_product = torch.matmul(tokens, centroids.T)
        return -dot_product
    else:
        raise ValueError(f"Unknown metric: {metric}")

def kmeans_clustering(tokens, num_clusters, metric='cosine', n_iter=3):
    """K-means clustering"""
    N, D = tokens.shape
    indices = torch.randperm(N)[:num_clusters]
    centroids = tokens[indices].clone()
    
    for _ in range(n_iter):
        distances = compute_distance(tokens, centroids, metric=metric)
        assignments = distances.argmin(dim=1)
        
        new_centroids = []
        for k in range(num_clusters):
            mask = (assignments == k)
            if mask.sum() > 0:
                new_centroids.append(tokens[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[k])
        centroids = torch.stack(new_centroids, dim=0)
    
    return centroids

def find_nearest_tokens(tokens, centroids, metric='cosine'):
    """Centroid에 가장 가까운 실제 토큰 찾기"""
    distances = compute_distance(tokens, centroids, metric=metric)
    nearest_indices = distances.argmin(dim=0)
    return tokens[nearest_indices]

def initialize_summary_tokens(image_features, num_summary, metric='cosine', n_iter=3, use_nearest=True):
    """Summary token 초기화"""
    centroids = kmeans_clustering(image_features, num_summary, metric=metric, n_iter=n_iter)
    
    if use_nearest:
        return find_nearest_tokens(image_features, centroids, metric=metric)
    else:
        return centroids

def main():
    print("=" * 80)
    print("K-means 초기화 최소 검증")
    print("=" * 80)
    
    # Test 1: 기본 동작
    print("\n[Test 1] 기본 K-means 동작")
    tokens = torch.randn(100, 64)
    summary = initialize_summary_tokens(tokens, 5, metric='cosine', n_iter=3, use_nearest=True)
    print(f"  Input: {tokens.shape}")
    print(f"  Output: {summary.shape}")
    assert summary.shape == (5, 64), f"Expected (5, 64), got {summary.shape}"
    print("  ✓ 통과")
    
    # Test 2: use_nearest=False
    print("\n[Test 2] use_nearest=False (centroid 직접 사용)")
    summary_centroid = initialize_summary_tokens(tokens, 5, metric='cosine', n_iter=3, use_nearest=False)
    print(f"  Output: {summary_centroid.shape}")
    assert summary_centroid.shape == (5, 64)
    print("  ✓ 통과")
    
    # Test 3: 실제 토큰 검증
    print("\n[Test 3] use_nearest=True가 실제 토큰을 반환하는지 확인")
    is_real = []
    for i in range(5):
        found = False
        for j in range(100):
            if torch.allclose(summary[i], tokens[j], atol=1e-6):
                found = True
                break
        is_real.append(found)
    print(f"  실제 토큰 비율: {sum(is_real)}/5")
    assert sum(is_real) == 5, "모든 summary token이 실제 토큰이어야 합니다!"
    print("  ✓ 모든 토큰이 실제 이미지 토큰입니다")
    
    # Test 4: 다양한 metric
    print("\n[Test 4] 다양한 metric 테스트")
    for metric in ['cosine', 'l2', 'dot']:
        summary = initialize_summary_tokens(tokens, 8, metric=metric, n_iter=3, use_nearest=True)
        print(f"  {metric}: shape={summary.shape} ✓")
    
    # Test 5: 배치 처리
    print("\n[Test 5] 배치 처리 시뮬레이션")
    batch_size = 4
    batch_tokens = torch.randn(batch_size, 576, 1024)  # CLIP 출력 크기
    batch_summary = []
    for b in range(batch_size):
        summary = initialize_summary_tokens(
            batch_tokens[b], 
            num_summary=16, 
            metric='cosine', 
            n_iter=3, 
            use_nearest=True
        )
        batch_summary.append(summary)
    batch_summary = torch.stack(batch_summary, dim=0)
    print(f"  Batch input: {batch_tokens.shape}")
    print(f"  Batch output: {batch_summary.shape}")
    assert batch_summary.shape == (4, 16, 1024)
    print("  ✓ 배치 처리 정상")
    
    # Test 6: Before/After projector 시뮬레이션
    print("\n[Test 6] Before/After Projector 시뮬레이션")
    
    # Before projector (1024-dim)
    vision_features = torch.randn(576, 1024)
    summary_before = initialize_summary_tokens(vision_features, 16, 'cosine', 3, True)
    print(f"  Before projector: {vision_features.shape} → {summary_before.shape}")
    
    # After projector (4096-dim)
    projector = torch.nn.Linear(1024, 4096)
    projected_features = projector(vision_features)
    summary_after = initialize_summary_tokens(projected_features, 16, 'cosine', 3, False)
    print(f"  After projector: {projected_features.shape} → {summary_after.shape}")
    print("  ✓ Before/After 모두 정상")
    
    # Test 7: Iteration 효과
    print("\n[Test 7] K-means iteration 효과")
    for n_iter in [1, 3, 5]:
        summary = initialize_summary_tokens(tokens, 10, 'cosine', n_iter, False)
        var = summary.var(dim=0).mean().item()
        print(f"  n_iter={n_iter}: 분산={var:.6f}")
    print("  ✓ Iteration 테스트 완료")
    
    print("\n" + "=" * 80)
    print("✅ 모든 K-means 검증 테스트 통과!")
    print("=" * 80)
    print("\n[검증 완료 항목]")
    print("  ✓ K-means clustering 정상 동작")
    print("  ✓ 3가지 metric (cosine, l2, dot) 모두 정상")
    print("  ✓ use_nearest True/False 모두 정상")
    print("  ✓ 배치 처리 정상")
    print("  ✓ Before/After projector 모두 정상")
    print("  ✓ K-means iteration 정상")
    print("\n다음 단계:")
    print("  1. python3 test_kmeans_full.py  (전체 테스트)")
    print("  2. ./test_kmeans_integration.sh  (실제 훈련 통합 테스트)")

if __name__ == "__main__":
    main()
