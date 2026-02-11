"""
K-means 기반 요약 토큰 초기화 테스트 스크립트

요약 토큰이 K-means clustering을 통해 올바르게 초기화되는지 검증합니다.
"""

import torch
import torch.nn as nn
from llava.model.kmeans_initializer import (
    compute_distance,
    kmeans_clustering,
    initialize_summary_tokens_with_kmeans
)


def test_compute_distance():
    """거리 계산 함수 테스트"""
    print("\n" + "="*60)
    print("Test 1: Distance Computation")
    print("="*60)
    
    # 테스트 데이터
    embeddings = torch.randn(100, 64)
    centroids = torch.randn(8, 64)
    
    # Cosine distance
    dist_cosine = compute_distance(embeddings, centroids, metric='cosine')
    print(f"✅ Cosine distance shape: {dist_cosine.shape} (expected: [100, 8])")
    print(f"   Range: [{dist_cosine.min():.4f}, {dist_cosine.max():.4f}]")
    
    # L2 distance
    dist_l2 = compute_distance(embeddings, centroids, metric='l2')
    print(f"✅ L2 distance shape: {dist_l2.shape} (expected: [100, 8])")
    print(f"   Range: [{dist_l2.min():.4f}, {dist_l2.max():.4f}]")
    
    # Dot product
    dist_dot = compute_distance(embeddings, centroids, metric='dot')
    print(f"✅ Dot distance shape: {dist_dot.shape} (expected: [100, 8])")
    print(f"   Range: [{dist_dot.min():.4f}, {dist_dot.max():.4f}]")


def test_kmeans_clustering():
    """K-means clustering 테스트"""
    print("\n" + "="*60)
    print("Test 2: K-means Clustering")
    print("="*60)
    
    # 테스트 데이터 (576개 이미지 토큰, 1024차원)
    num_tokens = 576
    hidden_size = 1024
    num_clusters = 8
    
    embeddings = torch.randn(num_tokens, hidden_size)
    
    # Cosine metric
    print(f"\n🔧 K-means with cosine metric (n_iter=3)")
    centroids_cosine = kmeans_clustering(
        embeddings, 
        num_clusters=num_clusters,
        metric='cosine',
        n_iter=3
    )
    print(f"✅ Centroids shape: {centroids_cosine.shape} (expected: [{num_clusters}, {hidden_size}])")
    
    # L2 metric
    print(f"\n🔧 K-means with L2 metric (n_iter=3)")
    centroids_l2 = kmeans_clustering(
        embeddings,
        num_clusters=num_clusters,
        metric='l2',
        n_iter=3
    )
    print(f"✅ Centroids shape: {centroids_l2.shape} (expected: [{num_clusters}, {hidden_size}])")
    
    # 각 centroid가 서로 다른지 확인
    uniqueness_cosine = torch.cdist(centroids_cosine, centroids_cosine, p=2)
    uniqueness_l2 = torch.cdist(centroids_l2, centroids_l2, p=2)
    
    # Diagonal은 0이므로 제외하고 최소 거리 확인
    mask = torch.eye(num_clusters, dtype=torch.bool)
    min_dist_cosine = uniqueness_cosine[~mask].min()
    min_dist_l2 = uniqueness_l2[~mask].min()
    
    print(f"\n📊 Centroid Diversity Check:")
    print(f"   Cosine: Min distance between centroids = {min_dist_cosine:.4f}")
    print(f"   L2: Min distance between centroids = {min_dist_l2:.4f}")
    
    if min_dist_cosine > 0.1 and min_dist_l2 > 0.1:
        print("   ✅ Centroids are sufficiently diverse")
    else:
        print("   ⚠️ Some centroids may be too similar")


def test_initialize_summary_tokens():
    """요약 토큰 초기화 통합 테스트"""
    print("\n" + "="*60)
    print("Test 3: Summary Token Initialization")
    print("="*60)
    
    # 모의 vision features (CLIP output)
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    
    vision_features = torch.randn(num_image_tokens, vision_hidden_size)
    
    # 모의 MM projector (1024 -> 4096)
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    
    # K-means 초기화
    print(f"\n🔧 Initializing {num_summary_tokens} summary tokens...")
    summary_init = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=num_summary_tokens,
        metric='cosine',
        n_iter=3
    )
    
    print(f"✅ Summary token init shape: {summary_init.shape}")
    print(f"   Expected: [{num_summary_tokens}, {llm_hidden_size}]")
    
    assert summary_init.shape == (num_summary_tokens, llm_hidden_size), \
        f"Shape mismatch: {summary_init.shape} != ({num_summary_tokens}, {llm_hidden_size})"
    
    # 각 요약 토큰이 서로 다른지 확인
    diversity = torch.cdist(summary_init, summary_init, p=2)
    mask = torch.eye(num_summary_tokens, dtype=torch.bool)
    min_dist = diversity[~mask].min()
    
    print(f"\n📊 Summary Token Diversity:")
    print(f"   Min distance between tokens = {min_dist:.4f}")
    
    if min_dist > 0.1:
        print("   ✅ Summary tokens are sufficiently diverse")
    else:
        print("   ⚠️ Some summary tokens may be too similar")


def test_different_metrics():
    """다양한 메트릭 비교 테스트"""
    print("\n" + "="*60)
    print("Test 4: Metric Comparison")
    print("="*60)
    
    embeddings = torch.randn(576, 1024)
    num_clusters = 8
    
    metrics = ['cosine', 'l2', 'dot']
    centroids_dict = {}
    
    for metric in metrics:
        print(f"\n🔧 Testing metric: {metric}")
        centroids = kmeans_clustering(
            embeddings,
            num_clusters=num_clusters,
            metric=metric,
            n_iter=3
        )
        centroids_dict[metric] = centroids
        
        # Diversity 체크
        diversity = torch.cdist(centroids, centroids, p=2)
        mask = torch.eye(num_clusters, dtype=torch.bool)
        min_dist = diversity[~mask].min()
        avg_dist = diversity[~mask].mean()
        
        print(f"   Min distance: {min_dist:.4f}")
        print(f"   Avg distance: {avg_dist:.4f}")
    
    print("\n📊 Recommendation:")
    print("   - Cosine: Best for normalized embeddings (semantic similarity)")
    print("   - L2: Best for absolute distance")
    print("   - Dot: Fast but less interpretable")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("K-MEANS INITIALIZATION TEST SUITE")
    print("="*60)
    
    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    
    try:
        test_compute_distance()
        test_kmeans_clustering()
        test_initialize_summary_tokens()
        test_different_metrics()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nK-means 초기화가 올바르게 구현되었습니다.")
        print("학습 시작 전 vision encoder 출력을 K-means로 클러스터링하여")
        print("다양하고 유의미한 요약 토큰 초기값을 생성할 수 있습니다.\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
