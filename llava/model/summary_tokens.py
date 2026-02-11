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


import torch
import torch.nn as nn


class SummaryTokens(nn.Module):
    """
    학습 가능한 이미지 요약 토큰을 관리하는 모듈
    
    이미지 정보를 압축하여 표현하는 학습 가능한 토큰들을 생성합니다.
    각 요약 토큰은 이미지의 서로 다른 측면을 포착하도록 학습됩니다.
    
    Args:
        num_summary_tokens (int): 요약 토큰의 개수 (1, 4, 8, 16, 32 등)
        hidden_size (int): 토큰의 임베딩 차원 (LLM의 hidden_size와 동일)
    """
    # 요약 토큰 개수
    # 임베딩 차원
    def __init__(self, num_summary_tokens: int, hidden_size: int):
        super().__init__()
        self.num_summary_tokens = num_summary_tokens
        self.hidden_size = hidden_size
        
        # 학습 가능한 요약 토큰 임베딩 초기화 (무작위 초기화)
        # Shape: [num_summary_tokens, hidden_size]
        self.summary_embeddings = nn.Parameter(
            torch.randn(num_summary_tokens, hidden_size)
        )
        
        # Xavier uniform 초기화로 안정적인 학습 시작
        nn.init.xavier_uniform_(self.summary_embeddings)
    
    def forward(self, batch_size: int = 1):
        """
        배치 크기에 맞게 요약 토큰 임베딩을 반환
        
        Args:
            batch_size (int): 배치 크기
            
        Returns:
            torch.Tensor: Shape [batch_size, num_summary_tokens, hidden_size]
        """
        # 배치 차원 확장: [num_summary_tokens, hidden_size] -> [batch_size, num_summary_tokens, hidden_size]
        return self.summary_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_single_batch(self):
        """
        단일 배치용 요약 토큰 임베딩 반환
        
        Returns:
            torch.Tensor: Shape [num_summary_tokens, hidden_size]
        """
        return self.summary_embeddings
    
    def initialize_from_image_tokens(self, image_token_embeddings: torch.Tensor):
        """
        이미지 토큰 임베딩의 평균값으로 요약 토큰 초기화
        (K-means 초기화가 비활성화된 경우 사용)
        
        Args:
            image_token_embeddings (torch.Tensor): 
                Shape [num_image_tokens, hidden_size]
        """
        assert image_token_embeddings.shape[1] == self.hidden_size, \
            f"Hidden size mismatch: {image_token_embeddings.shape[1]} vs {self.hidden_size}"
        
        # 간단한 평균 기반 초기화
        num_image_tokens = image_token_embeddings.shape[0]
        
        if num_image_tokens >= self.num_summary_tokens:
            # 이미지 토큰을 균등하게 나눠서 각 그룹의 평균을 요약 토큰으로 설정
            chunk_size = num_image_tokens // self.num_summary_tokens
            for i in range(self.num_summary_tokens):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.num_summary_tokens - 1 else num_image_tokens
                self.summary_embeddings.data[i] = image_token_embeddings[start_idx:end_idx].mean(dim=0)
        else:
            # 이미지 토큰이 요약 토큰보다 적은 경우, 반복해서 사용
            for i in range(self.num_summary_tokens):
                self.summary_embeddings.data[i] = image_token_embeddings[i % num_image_tokens]
    
    def initialize_from_centroids(self, centroids: torch.Tensor):
        """
        K-means centroid로 요약 토큰 초기화
        
        Args:
            centroids (torch.Tensor): 
                Shape [num_summary_tokens, hidden_size]
                K-means로 추출된 centroid들
        """
        assert centroids.shape[0] == self.num_summary_tokens, \
            f"Number of centroids mismatch: {centroids.shape[0]} vs {self.num_summary_tokens}"
        assert centroids.shape[1] == self.hidden_size, \
            f"Hidden size mismatch: {centroids.shape[1]} vs {self.hidden_size}"
        
        # Centroid를 요약 토큰 임베딩으로 복사
        self.summary_embeddings.data.copy_(centroids)


def build_summary_tokens(config):
    """
    Config를 기반으로 SummaryTokens 모듈 생성
    
    Args:
        config: 모델 설정 객체
        
    Returns:
        SummaryTokens: 요약 토큰 모듈
    """
    # config에서 num_summary_tokens와 hidden_size 추출
    # 정의가 안되어 있을 경우 32와 4096을 기본값으로 사용
    # 4096은 LLaMA 모델의 hidden_size 기본값
    num_summary_tokens = getattr(config, 'num_summary_tokens', 32)  # 기본값: 32개
    hidden_size = getattr(config, 'hidden_size', 4096) # LLM hidden size 기본값: 4096
    
    return SummaryTokens(num_summary_tokens, hidden_size)
