#    Copyright 2024
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 확인 필요

import torch
import torch.nn.functional as F
from typing import Literal


def compute_distance(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    metric: Literal['cosine', 'l2', 'dot'] = 'cosine'
) -> torch.Tensor:
    """
    임베딩과 centroid 간의 거리/유사도 계산
    
    Args:
        embeddings: [N, D] 임베딩 벡터들
        centroids: [K, D] centroid 벡터들
        metric: 거리 메트릭 ('cosine', 'l2', 'dot')
        
    Returns:
        distances: [N, K] 거리 행렬 (작을수록 가까움)
    """
    if metric == 'cosine':
        # 코사인 유사도: 1 - cosine_similarity
        # L2 정규화
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        centroids_norm = F.normalize(centroids, p=2, dim=1)
        # 코사인 유사도 계산 (높을수록 유사)
        similarity = torch.mm(embeddings_norm, centroids_norm.t())
        # 거리로 변환 (낮을수록 유사)
        distances = 1 - similarity
        
    elif metric == 'l2':
        # L2 거리 (유클리드 거리)
        # [N, 1, D] - [1, K, D] = [N, K, D]
        diff = embeddings.unsqueeze(1) - centroids.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=2)
        
    elif metric == 'dot':
        # 음의 내적 (내적이 클수록 유사하므로 음수로 변환)
        similarity = torch.mm(embeddings, centroids.t())
        distances = -similarity
        
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine', 'l2', or 'dot'.")
    
    return distances


def kmeans_clustering(
    embeddings: torch.Tensor,
    num_clusters: int,
    metric: Literal['cosine', 'l2', 'dot'] = 'cosine',
    n_iter: int = 3,
    random_state: int = 42
) -> torch.Tensor:
    """
    K-means clustering을 사용하여 임베딩을 클러스터링하고 centroid 반환
    
    Args:
        embeddings: [N, D] 임베딩 벡터들
        num_clusters: 클러스터 개수 (K)
        metric: 거리 메트릭 ('cosine', 'l2', 'dot')
        n_iter: K-means 반복 횟수
        random_state: 재현성을 위한 시드
        
    Returns:
        centroids: [K, D] 클러스터 centroid들
    """
    N, D = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype
    
    # 시드 설정
    torch.manual_seed(random_state)
    
    # 초기 centroid 무작위 선택 (K개의 임베딩을 무작위로 선택)
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = embeddings[indices].clone()  # [K, D]
    
    # K-means 반복
    for iteration in range(n_iter):
        # 1. 각 임베딩을 가장 가까운 centroid에 할당
        distances = compute_distance(embeddings, centroids, metric)  # [N, K]
        assignments = torch.argmin(distances, dim=1)  # [N]
        
        # 2. 각 클러스터의 centroid 업데이트
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = (assignments == k)
            if mask.sum() > 0:
                # 해당 클러스터에 속한 임베딩들의 평균
                cluster_embeddings = embeddings[mask]
                new_centroids[k] = cluster_embeddings.mean(dim=0)
            else:
                # 빈 클러스터는 이전 centroid 유지
                new_centroids[k] = centroids[k]
        
        centroids = new_centroids
        
        # 코사인 메트릭 사용 시 centroid 정규화
        if metric == 'cosine':
            centroids = F.normalize(centroids, p=2, dim=1)
    
    return centroids


def find_nearest_tokens_to_centroids(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    metric: Literal['cosine', 'l2', 'dot'] = 'cosine'
) -> torch.Tensor:
    """
    각 centroid와 가장 가까운 실제 임베딩(토큰) 찾기
    
    Args:
        embeddings: [N, D] 원본 임베딩들
        centroids: [K, D] K-means centroid들
        metric: 거리 메트릭
        
    Returns:
        nearest_tokens: [K, D] 각 centroid에 가장 가까운 실제 토큰들
    """
    # 각 centroid에 대해 가장 가까운 임베딩 찾기
    distances = compute_distance(embeddings, centroids, metric)  # [N, K]
    
    # 각 centroid(열)에 대해 가장 가까운 임베딩(행) 인덱스 찾기
    nearest_indices = torch.argmin(distances, dim=0)  # [K]
    
    # 가장 가까운 실제 토큰들 추출
    nearest_tokens = embeddings[nearest_indices]  # [K, D]
    
    return nearest_tokens


def initialize_summary_tokens_with_kmeans(
    vision_features: torch.Tensor,
    mm_projector: torch.nn.Module,
    num_summary_tokens: int,
    metric: Literal['cosine', 'l2', 'dot'] = 'cosine',
    n_iter: int = 3,
    random_state: int = 42,
    use_nearest: bool = True
) -> torch.Tensor:
    """
    K-means clustering으로 요약 토큰 초기값 생성
    
    워크플로우:
    1. Vision encoder 출력 → K-means clustering
    2. Centroid 추출 (또는 가장 가까운 실제 토큰)
    3. MM Projector 통과 → 요약 토큰 초기값
    
    Args:
        vision_features: Vision encoder 출력 [N, vision_hidden_size]
        mm_projector: 멀티모달 프로젝터 (vision_hidden_size → llm_hidden_size)
        num_summary_tokens: 요약 토큰 개수
        metric: K-means 거리 메트릭 ('cosine', 'l2', 'dot')
        n_iter: K-means 반복 횟수
        random_state: 재현성을 위한 시드
        use_nearest: True면 centroid와 가장 가까운 실제 토큰 사용, False면 centroid 직접 사용
        
    Returns:
        summary_token_init: [num_summary_tokens, llm_hidden_size] 요약 토큰 초기값
    """
    with torch.no_grad():
        # Vision features가 배치 차원을 가지면 제거
        if vision_features.dim() == 3:
            # [1, N, D] → [N, D]
            vision_features = vision_features.squeeze(0)
        
        # mm_projector의 dtype 확인 및 vision_features dtype 맞추기
        projector_dtype = next(mm_projector.parameters()).dtype
        vision_features = vision_features.to(dtype=projector_dtype)
        
        # K-means clustering으로 centroid 추출
        centroids = kmeans_clustering(
            embeddings=vision_features,
            num_clusters=num_summary_tokens,
            metric=metric,
            n_iter=n_iter,
            random_state=random_state
        )  # [num_summary_tokens, vision_hidden_size]
        
        # 대표 토큰 선택
        if use_nearest:
            # Centroid와 가장 가까운 실제 이미지 토큰 사용
            representative_tokens = find_nearest_tokens_to_centroids(
                embeddings=vision_features,
                centroids=centroids,
                metric=metric
            )  # [num_summary_tokens, vision_hidden_size]
        else:
            # Centroid를 직접 사용
            representative_tokens = centroids
        
        # 대표 토큰을 MM Projector에 통과
        # Projector는 보통 배치 차원을 기대하므로 unsqueeze
        if representative_tokens.dim() == 2:
            representative_tokens = representative_tokens.unsqueeze(0)  # [1, num_summary_tokens, vision_hidden_size]
        
        summary_token_init = mm_projector(representative_tokens)  # [1, num_summary_tokens, llm_hidden_size]
        
        # 배치 차원 제거
        summary_token_init = summary_token_init.squeeze(0)  # [num_summary_tokens, llm_hidden_size]
    
    return summary_token_init


def initialize_model_summary_tokens(
    model,
    sample_images: torch.Tensor,
    metric: Literal['cosine', 'l2', 'dot'] = 'cosine',
    n_iter: int = 3,
    random_state: int = 42
):
    """
    모델의 요약 토큰을 K-means 기반으로 초기화
    
    Args:
        model: LLaVA 모델 (vision_tower, mm_projector, summary_tokens 포함)
        sample_images: 샘플 이미지 텐서 [1, C, H, W] 또는 [C, H, W]
        metric: K-means 거리 메트릭
        n_iter: K-means 반복 횟수
        random_state: 재현성을 위한 시드
    """
    vision_tower = model.get_vision_tower()
    mm_projector = model.get_model().mm_projector
    summary_tokens_module = model.get_summary_tokens()
    
    if vision_tower is None or mm_projector is None or summary_tokens_module is None:
        raise ValueError("Model must have vision_tower, mm_projector, and summary_tokens")
    
    # 이미지 배치 차원 확인
    if sample_images.dim() == 3:
        sample_images = sample_images.unsqueeze(0)  # [1, C, H, W]
    
    # Vision encoder를 통과하여 이미지 특징 추출
    with torch.no_grad():
        if hasattr(model, 'encode_images'):
            # LLaVA의 encode_images 사용
            vision_features = model.encode_images(sample_images)
            if isinstance(vision_features, list):
                vision_features = vision_features[0]
        else:
            # 직접 vision tower 사용
            vision_features = vision_tower(sample_images)
            if isinstance(vision_features, (list, tuple)):
                vision_features = vision_features[0]
    
    # K-means 초기화 수행
    summary_token_init = initialize_summary_tokens_with_kmeans(
        vision_features=vision_features,
        mm_projector=mm_projector,
        num_summary_tokens=summary_tokens_module.num_summary_tokens,
        metric=metric,
        n_iter=n_iter,
        random_state=random_state
    )
    
    # 요약 토큰 파라미터 업데이트
    summary_tokens_module.summary_embeddings.data.copy_(summary_token_init)
    
    print(f"✅ Summary tokens initialized with K-means (metric={metric}, n_iter={n_iter})")
    print(f"   Shape: {summary_token_init.shape}")
