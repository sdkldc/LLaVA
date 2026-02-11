"""
K-means 초기화 빠른 검증 스크립트
- 모델 로딩 없이 순수 K-means 로직만 빠르게 테스트
- llava_arch.py의 실제 동작 시뮬레이션
"""

import os
# DeepSpeed 비활성화
os.environ['DEEPSPEED_DISABLE'] = '1'

import torch
import sys

# 직접 import하지 않고 경로 추가
sys.path.insert(0, '/home/deokhyeon/Documents/LLaVA')

from llava.model.kmeans_initializer import initialize_summary_tokens_with_kmeans

def simulate_forward_pass():
    """실제 forward pass에서의 K-means 동작 시뮬레이션"""
    print("=" * 80)
    print("K-means 초기화 Forward Pass 시뮬레이션")
    print("=" * 80)
    
    # 시뮬레이션 설정
    batch_size = 2
    num_patches = 576  # 24x24 CLIP patches
    vision_dim = 1024  # CLIP hidden dim
    projector_dim = 4096  # LLaMA hidden dim
    num_summary = 16
    
    print(f"\n[설정]")
    print(f"  Batch size: {batch_size}")
    print(f"  Image patches: {num_patches}")
    print(f"  Vision dim: {vision_dim}")
    print(f"  Projector dim: {projector_dim}")
    print(f"  Summary tokens: {num_summary}")
    
    # 1. Vision Encoder 출력 시뮬레이션
    print(f"\n[Step 1] Vision Encoder 출력")
    vision_features = torch.randn(batch_size, num_patches, vision_dim)
    print(f"  Shape: {vision_features.shape}")
    
    # 2. Before Projector K-means 테스트
    print(f"\n[Step 2] K-means (before_projector)")
    print(f"  → Vision space (1024-dim)에서 clustering")
    
    # 더미 프로젝터 생성
    mm_projector = torch.nn.Linear(1024, 4096)
    
    batch_summary_before = []
    for b in range(batch_size):
        summary = initialize_summary_tokens_with_kmeans(
            vision_features=vision_features[b],  # [576, 1024]
            mm_projector=mm_projector,
            num_summary_tokens=num_summary,
            metric='cosine',
            n_iter=3,
            use_nearest=True
        )
        batch_summary_before.append(summary)
        print(f"  Batch {b}: {summary.shape}")
    
    batch_summary_before = torch.stack(batch_summary_before, dim=0)
    print(f"  ✓ Final shape: {batch_summary_before.shape}")
    
    # 3. Projector 시뮬레이션
    print(f"\n[Step 3] MM Projector 통과")
    # 실제로는 MLP이지만 여기서는 단순 linear로 시뮬레이션
    projector = torch.nn.Linear(vision_dim, projector_dim)
    vision_features_projected = projector(vision_features)
    print(f"  Shape: {vision_features_projected.shape}")
    
    # 4. After Projector K-means 테스트
    print(f"\n[Step 4] K-means (after_projector)")
    print(f"  → LLM space (4096-dim)에서 clustering")
    
    # After projector의 경우 이미 projected된 상태
    # Identity projector 사용 (입력 그대로 출력)
    identity_projector = torch.nn.Identity()
    
    batch_summary_after = []
    for b in range(batch_size):
        # after_projector의 경우: 이미 projector를 거친 features에서 직접 clustering
        # 여기서는 kmeans 결과를 바로 사용 (projector 불필요)
        # 하지만 API 일관성을 위해 identity 사용
        from llava.model.kmeans_initializer import kmeans_clustering
        summary = kmeans_clustering(
            embeddings=vision_features_projected[b],  # [576, 4096]
            num_clusters=num_summary,
            metric='cosine',
            n_iter=3
        )
        batch_summary_after.append(summary)
        print(f"  Batch {b}: {summary.shape}")
    
    batch_summary_after = torch.stack(batch_summary_after, dim=0)
    print(f"  ✓ Final shape: {batch_summary_after.shape}")
    
    # 5. 결과 비교
    print(f"\n[Step 5] Before vs After 비교")
    
    # Before는 이미 projector를 거친 상태 (4096차원)
    # 직접 비교 가능
    batch_summary_before_projected = batch_summary_before
    
    # 코사인 유사도 계산
    def cosine_similarity(a, b):
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=-1).mean()
    
    for b in range(batch_size):
        sim = cosine_similarity(
            batch_summary_before_projected[b], 
            batch_summary_after[b]
        )
        print(f"  Batch {b} - Before와 After 코사인 유사도: {sim:.4f}")
    
    print(f"\n[Step 6] use_nearest True vs False 비교")
    
    # 같은 입력으로 두 방식 비교
    test_features = vision_features[0]  # [576, 1024]
    
    summary_centroid = initialize_summary_tokens_with_kmeans(
        vision_features=test_features,
        mm_projector=projector,
        num_summary_tokens=num_summary,
        metric='cosine',
        n_iter=3,
        use_nearest=False
    )
    
    summary_nearest = initialize_summary_tokens_with_kmeans(
        vision_features=test_features,
        mm_projector=projector,
        num_summary_tokens=num_summary,
        metric='cosine',
        n_iter=3,
        use_nearest=True
    )
    
    # nearest가 실제 토큰인지 확인하려면 projection 전 비교 필요
    # 여기서는 출력 shape만 확인
    print(f"  Centroid 방식: shape={summary_centroid.shape}")
    print(f"  Nearest 방식: shape={summary_nearest.shape}")
    
    # 두 방식의 차이
    diff = (summary_centroid - summary_nearest).abs().mean()
    print(f"  Centroid vs Nearest 평균 차이: {diff:.6f}")
    
    return True

def test_different_metrics():
    """다양한 metric 비교"""
    print("\n" + "=" * 80)
    print("Metric 비교 테스트")
    print("=" * 80)
    
    features = torch.randn(576, 1024)
    num_summary = 8
    projector = torch.nn.Linear(1024, 4096)
    
    results = {}
    for metric in ['cosine', 'l2', 'dot']:
        print(f"\n[{metric.upper()}]")
        summary = initialize_summary_tokens_with_kmeans(
            vision_features=features,
            mm_projector=projector,
            num_summary_tokens=num_summary,
            metric=metric,
            n_iter=3,
            use_nearest=True
        )
        
        # 선택된 토큰들의 분산 계산
        var = summary.var(dim=0).mean().item()
        norm = summary.norm(dim=-1).mean().item()
        
        print(f"  Shape: {summary.shape}")
        print(f"  분산: {var:.6f}")
        print(f"  평균 norm: {norm:.6f}")
        
        results[metric] = {'var': var, 'norm': norm}
    
    print(f"\n[비교 결과]")
    for metric, stats in results.items():
        print(f"  {metric}: var={stats['var']:.6f}, norm={stats['norm']:.6f}")
    
    return True

def test_iteration_effect():
    """K-means 반복 횟수 효과 테스트"""
    print("\n" + "=" * 80)
    print("K-means 반복 횟수 효과 테스트")
    print("=" * 80)
    
    features = torch.randn(576, 1024)
    num_summary = 16
    projector = torch.nn.Linear(1024, 4096)
    
    prev_summary = None
    for n_iter in [1, 3, 5, 10]:
        summary = initialize_summary_tokens_with_kmeans(
            vision_features=features,
            mm_projector=projector,
            num_summary_tokens=num_summary,
            metric='cosine',
            n_iter=n_iter,
            use_nearest=False
        )
        
        print(f"\n[n_iter={n_iter}]")
        print(f"  Shape: {summary.shape}")
        
        if prev_summary is not None:
            # 이전 결과와 비교
            diff = (summary - prev_summary).abs().mean().item()
            print(f"  이전 iteration과의 차이: {diff:.6f}")
        
        prev_summary = summary.clone()
    
    return True

def main():
    print("\n")
    
    try:
        # Test 1: Forward pass 시뮬레이션
        simulate_forward_pass()
        
        # Test 2: Metric 비교
        test_different_metrics()
        
        # Test 3: Iteration 효과
        test_iteration_effect()
        
        print("\n" + "=" * 80)
        print("✅ 모든 빠른 검증 테스트 통과!")
        print("=" * 80)
        print("\n[검증 완료 항목]")
        print("  ✓ Before/After projector K-means 정상 동작")
        print("  ✓ use_nearest True/False 정상 동작")
        print("  ✓ 3가지 metric 모두 정상")
        print("  ✓ K-means iteration 효과 확인")
        print("\n다음 단계: python test_kmeans_full.py 실행")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
