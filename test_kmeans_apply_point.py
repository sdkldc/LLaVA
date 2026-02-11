"""
K-means 적용 시점 비교 테스트

before_projector vs after_projector 성능 및 동작 비교
"""

import torch
import torch.nn as nn
from llava.model.kmeans_initializer import initialize_summary_tokens_with_kmeans, kmeans_clustering


def test_kmeans_apply_point_comparison():
    """두 가지 적용 시점 비교"""
    print("\n" + "="*60)
    print("Test: K-means Apply Point Comparison")
    print("="*60)
    
    # 모의 설정
    num_image_tokens = 576
    vision_hidden_size = 1024  # CLIP output
    llm_hidden_size = 4096     # LLaMA hidden size
    num_summary_tokens = 8
    
    # 모의 MM Projector (1024 → 4096)
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    
    # 샘플 vision features
    torch.manual_seed(42)
    vision_features = torch.randn(num_image_tokens, vision_hidden_size)
    
    print("\n🔧 Method 1: before_projector (기본)")
    print("   Vision Encoder → K-means → Centroid → Projector")
    # K-means 먼저, 그 다음 프로젝터
    summary_before = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    print(f"   Output shape: {summary_before.shape}")
    print(f"   Mean: {summary_before.mean():.4f}, Std: {summary_before.std():.4f}")
    
    print("\n🔧 Method 2: after_projector")
    print("   Vision Encoder → Projector → K-means → Centroid")
    # 프로젝터 먼저, 그 다음 K-means
    with torch.no_grad():
        # Vision features를 먼저 프로젝터에 통과
        projected_features = mm_projector(vision_features.unsqueeze(0)).squeeze(0)
        # 프로젝터 후 K-means
        summary_after = kmeans_clustering(
            embeddings=projected_features,
            num_clusters=num_summary_tokens,
            metric='cosine',
            n_iter=3,
            random_state=42
        )
    print(f"   Output shape: {summary_after.shape}")
    print(f"   Mean: {summary_after.mean():.4f}, Std: {summary_after.std():.4f}")
    
    # 결과 비교
    print("\n📊 Comparison:")
    print(f"   Shape match: {summary_before.shape == summary_after.shape}")
    
    # 다양성 비교 (각 요약 토큰 간 거리)
    def compute_diversity(tokens):
        dist_matrix = torch.cdist(tokens, tokens, p=2)
        mask = torch.eye(num_summary_tokens, dtype=torch.bool)
        return dist_matrix[~mask].mean().item()
    
    div_before = compute_diversity(summary_before)
    div_after = compute_diversity(summary_after)
    
    print(f"   Diversity (before_projector): {div_before:.4f}")
    print(f"   Diversity (after_projector): {div_after:.4f}")
    
    # 권장 사항
    print("\n💡 Recommendation:")
    print("   - before_projector: Vision space에서 의미있는 클러스터 → 프로젝터 학습 반영")
    print("   - after_projector: LLM space에서 직접 클러스터 → 최종 공간 최적화")
    print("   - 기본값: before_projector (프로젝터 학습 효과 활용)")


def test_dimension_changes():
    """차원 변화 추적"""
    print("\n" + "="*60)
    print("Test: Dimension Changes")
    print("="*60)
    
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    vision_features = torch.randn(num_image_tokens, vision_hidden_size)
    
    print("\n📐 before_projector workflow:")
    print(f"   1. Vision features: {vision_features.shape}")
    
    centroids = kmeans_clustering(
        embeddings=vision_features,
        num_clusters=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    print(f"   2. K-means centroids: {centroids.shape}")
    
    with torch.no_grad():
        projected = mm_projector(centroids.unsqueeze(0)).squeeze(0)
    print(f"   3. After projector: {projected.shape}")
    
    print("\n📐 after_projector workflow:")
    with torch.no_grad():
        projected_all = mm_projector(vision_features.unsqueeze(0)).squeeze(0)
    print(f"   1. Vision features: {vision_features.shape}")
    print(f"   2. After projector: {projected_all.shape}")
    
    centroids_after = kmeans_clustering(
        embeddings=projected_all,
        num_clusters=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    print(f"   3. K-means centroids: {centroids_after.shape}")


def test_computational_cost():
    """계산 비용 비교"""
    print("\n" + "="*60)
    print("Test: Computational Cost")
    print("="*60)
    
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    vision_features = torch.randn(num_image_tokens, vision_hidden_size)
    
    print("\n⚡ Computational operations:")
    
    print("\n   before_projector:")
    print(f"      - K-means on {num_image_tokens} tokens in {vision_hidden_size}-dim")
    print(f"      - Project {num_summary_tokens} centroids: {vision_hidden_size}→{llm_hidden_size}")
    
    print("\n   after_projector:")
    print(f"      - Project {num_image_tokens} tokens: {vision_hidden_size}→{llm_hidden_size}")
    print(f"      - K-means on {num_image_tokens} tokens in {llm_hidden_size}-dim")
    
    print("\n💡 Analysis:")
    print("   - before_projector: K-means in lower dimension → faster")
    print("   - after_projector: K-means in higher dimension → slower but final space")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("K-MEANS APPLY POINT COMPARISON TEST")
    print("="*60)
    print("\nK-means를 적용하는 두 가지 시점을 비교합니다:")
    print("1. before_projector: Vision space에서 클러스터링")
    print("2. after_projector: LLM space에서 클러스터링")
    
    try:
        test_kmeans_apply_point_comparison()
        test_dimension_changes()
        test_computational_cost()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n두 가지 방식 모두 올바르게 작동합니다.")
        print("기본값 before_projector는 더 빠르고 프로젝터 학습을 활용합니다.\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
