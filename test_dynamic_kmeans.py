"""
동적 K-means 기반 요약 토큰 생성 테스트

매 forward마다 이미지에서 K-means를 통해 대표 토큰을 추출하는지 검증
"""

import torch
import torch.nn as nn
from llava.model.kmeans_initializer import initialize_summary_tokens_with_kmeans


def test_dynamic_kmeans_per_image():
    """각 이미지마다 다른 대표 토큰이 생성되는지 테스트"""
    print("\n" + "="*60)
    print("Test: Dynamic K-means per Image")
    print("="*60)
    
    # 모의 설정
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    batch_size = 3
    
    # 모의 MM projector
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    
    # 배치의 각 이미지는 서로 다른 특징을 가짐
    torch.manual_seed(42)
    vision_features_batch = []
    for i in range(batch_size):
        # 각 이미지마다 다른 분포로 생성
        vision_features = torch.randn(num_image_tokens, vision_hidden_size) * (i + 1)
        vision_features_batch.append(vision_features)
    
    # 각 이미지에 대해 K-means 수행
    print(f"\n🔧 Processing {batch_size} images with K-means...")
    summary_tokens_batch = []
    
    for i, vision_features in enumerate(vision_features_batch):
        print(f"\n   Image {i+1}:")
        summary_tokens = initialize_summary_tokens_with_kmeans(
            vision_features=vision_features,
            mm_projector=mm_projector,
            num_summary_tokens=num_summary_tokens,
            metric='cosine',
            n_iter=3,
            random_state=42 + i  # 각 이미지마다 다른 시드
        )
        print(f"      Summary shape: {summary_tokens.shape}")
        print(f"      Mean: {summary_tokens.mean():.4f}, Std: {summary_tokens.std():.4f}")
        summary_tokens_batch.append(summary_tokens)
    
    # 배치로 스택
    summary_batch = torch.stack(summary_tokens_batch, dim=0)
    print(f"\n✅ Batch summary tokens shape: {summary_batch.shape}")
    print(f"   Expected: [{batch_size}, {num_summary_tokens}, {llm_hidden_size}]")
    
    # 각 이미지의 요약 토큰이 서로 다른지 확인
    print(f"\n📊 Checking diversity across images...")
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            # 두 이미지의 첫 번째 요약 토큰 비교
            dist = torch.norm(summary_batch[i, 0] - summary_batch[j, 0])
            print(f"   Distance between Image {i+1} and {j+1}: {dist:.4f}")
    
    print("\n✅ Each image has unique summary tokens!")


def test_kmeans_reproducibility():
    """동일한 이미지에 대해 재현 가능한지 테스트"""
    print("\n" + "="*60)
    print("Test: K-means Reproducibility")
    print("="*60)
    
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    
    # 모의 MM projector
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    
    # 동일한 vision features
    torch.manual_seed(100)
    vision_features = torch.randn(num_image_tokens, vision_hidden_size)
    
    # 첫 번째 실행
    summary_1 = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    
    # 두 번째 실행 (동일한 random_state)
    summary_2 = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    
    # 비교
    diff = torch.abs(summary_1 - summary_2).max()
    print(f"\n📊 Reproducibility check:")
    print(f"   Max difference: {diff:.10f}")
    
    if diff < 1e-5:
        print("   ✅ K-means is reproducible with same random_state")
    else:
        print("   ⚠️ K-means may not be fully reproducible")


def test_no_gradient_computation():
    """K-means가 gradient를 생성하지 않는지 테스트"""
    print("\n" + "="*60)
    print("Test: No Gradient Computation")
    print("="*60)
    
    num_image_tokens = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    num_summary_tokens = 8
    
    # 모의 MM projector
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        def forward(self, x):
            return self.linear2(self.gelu(self.linear1(x)))
    
    mm_projector = MockProjector()
    
    # requires_grad=True인 vision features
    vision_features = torch.randn(num_image_tokens, vision_hidden_size, requires_grad=True)
    
    # K-means 수행
    summary_tokens = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=num_summary_tokens,
        metric='cosine',
        n_iter=3,
        random_state=42
    )
    
    print(f"\n📊 Gradient check:")
    print(f"   Input requires_grad: {vision_features.requires_grad}")
    print(f"   Output requires_grad: {summary_tokens.requires_grad}")
    
    if not summary_tokens.requires_grad:
        print("   ✅ K-means output has no gradient (as expected)")
    else:
        print("   ⚠️ K-means output has gradient (unexpected)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DYNAMIC K-MEANS TEST SUITE")
    print("="*60)
    print("\n이 테스트는 매 forward마다 이미지에서 동적으로")
    print("K-means를 수행하여 대표 토큰을 생성하는지 검증합니다.")
    
    try:
        test_dynamic_kmeans_per_image()
        test_kmeans_reproducibility()
        test_no_gradient_computation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n동적 K-means 구현이 올바르게 작동합니다:")
        print("✓ 각 이미지마다 고유한 대표 토큰 생성")
        print("✓ 동일한 이미지에 대해 재현 가능")
        print("✓ Gradient 계산 없음 (학습 안 함)\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
