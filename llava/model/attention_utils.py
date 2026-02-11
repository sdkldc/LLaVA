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


def create_summary_token_attention_mask(
    batch_size: int,
    seq_length: int,
    summary_token_positions: tuple,
    device: torch.device,
    dtype: torch.dtype = torch.float32
):
    """
    요약 토큰 간 상호 참조를 방지하는 커스텀 Attention Mask 생성
    
    첫 번째 forward에서 사용:
    - 요약 토큰들이 서로를 참조하지 못하도록 마스킹
    - 각 요약 토큰은 이미지와 프롬프트만 참조 가능
    - 자기 자신은 참조 가능 (diagonal)
    
    Args:
        batch_size (int): 배치 크기
        seq_length (int): 전체 시퀀스 길이 (프롬프트 + 이미지 + 요약 토큰)
        summary_token_positions (tuple): (start_idx, end_idx) 요약 토큰 위치
        device (torch.device): 디바이스
        dtype (torch.dtype): 데이터 타입
        
    Returns:
        torch.Tensor: Attention mask [batch_size, 1, seq_length, seq_length]
                     False(0) = attend 가능, True(1) = attend 불가 (mask)
    """
    start_idx, end_idx = summary_token_positions
    num_summary_tokens = end_idx - start_idx
    
    # 기본 causal mask 생성 (lower triangular)
    # LLaMA의 causal attention을 유지
    # False(0) = 참조 가능, True(1) = 참조 불가
    causal_mask = torch.triu(
        torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
        diagonal=1
    )  # [seq_length, seq_length]
    
    # 요약 토큰 영역에서 서로 참조 방지
    # 요약 토큰 i -> 요약 토큰 j (i != j) 참조 차단
    for i in range(start_idx, end_idx):
        for j in range(start_idx, end_idx):
            if i != j:  # 자기 자신은 참조 가능
                causal_mask[i, j] = True  # 차단
    
    # 배치 차원 추가: [seq_length, seq_length] -> [batch_size, 1, seq_length, seq_length]
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    return attention_mask

# 최적화 버전, 더 빠름
def create_summary_token_attention_mask_optimized(
    batch_size: int,
    seq_length: int,
    summary_token_positions: tuple,
    device: torch.device,
    dtype: torch.dtype = torch.float32
):
    """
    최적화된 버전: 벡터화 연산으로 성능 향상
    
    Args:
        batch_size (int): 배치 크기
        seq_length (int): 전체 시퀀스 길이
        summary_token_positions (tuple): (start_idx, end_idx) 요약 토큰 위치
        device (torch.device): 디바이스
        dtype (torch.dtype): 데이터 타입
        
    Returns:
        torch.Tensor: Attention mask [batch_size, 1, seq_length, seq_length]
    """
    # 요약 토큰 워치
    start_idx, end_idx = summary_token_positions
    num_summary_tokens = end_idx - start_idx
    
    # 기본 causal mask
    # False(0) = 참조 가능, True(1) = 참조 불가
    # triu: upper triangular 부분을 1로 설정
    causal_mask = torch.triu(
        torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
        diagonal=1
    )
    
    # 요약 토큰 행 영역만 추출하여 수정
    # summary_region: [num_summary_tokens, seq_length]

    summary_region = causal_mask[start_idx:end_idx, :]
    
    # 요약 토큰끼리의 참조를 차단 (diagonal 제외)
    # 요약 토큰 영역에 대해 off-diagonal 요소를 True로 설정
    summary_cross_mask = torch.ones(
        num_summary_tokens, num_summary_tokens, 
        dtype=torch.bool, device=device
    )
    summary_cross_mask.fill_diagonal_(False)  # 자기 자신은 참조 가능
    
    # 요약 토큰 영역에 적용
    summary_region[:, start_idx:end_idx] = torch.logical_or(
        summary_region[:, start_idx:end_idx],
        summary_cross_mask
    )
    
    # 원본 mask에 반영
    causal_mask[start_idx:end_idx, :] = summary_region
    
    # 배치 차원 추가
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    return attention_mask


def visualize_attention_mask(attention_mask: torch.Tensor, save_path: str = None):
    """
    Attention mask를 시각화하여 디버깅
    
    Args:
        attention_mask (torch.Tensor): [batch_size, 1, seq_len, seq_len] or [seq_len, seq_len]
        save_path (str, optional): 저장 경로. None이면 표시만
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 2D로 변환
        if attention_mask.dim() == 4:
            mask_2d = attention_mask[0, 0].cpu().numpy()
        elif attention_mask.dim() == 2:
            mask_2d = attention_mask.cpu().numpy()
        else:
            raise ValueError(f"Unexpected attention mask shape: {attention_mask.shape}")
        
        # 시각화
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_2d, cmap='RdYlGn_r', interpolation='nearest')
        plt.colorbar(label='Mask Value (1=Blocked, 0=Allowed)')
        plt.title('Attention Mask Visualization')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention mask saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping visualization")


def combine_masks(
    custom_mask: torch.Tensor,
    padding_mask: torch.Tensor = None
):
    """
    커스텀 마스크(요약 토큰 간 참조 방지)와 패딩 마스크를 결합
    
    Args:
        custom_mask (torch.Tensor): Boolean mask [batch, 1, seq_len, seq_len]
                                    True = blocked, False = allowed
        padding_mask (torch.Tensor, optional): Padding mask [batch, seq_len]
                                               True = valid token, False = padding
                                               
    Returns:
        torch.Tensor: Combined boolean mask [batch, 1, seq_len, seq_len]
    """
    if padding_mask is None:
        return custom_mask
    
    # padding_mask를 4D로 확장: [batch, seq_len] -> [batch, 1, 1, seq_len]
    # Key 위치에서 패딩된 토큰은 참조 불가
    padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    
    # 패딩된 위치를 차단 (False -> True 변환)
    # padding_mask: True(유효), False(패딩)
    # attention mask: True(차단), False(허용)
    padding_block_mask = ~padding_mask_4d  # [batch, 1, 1, seq_len]
    
    # 두 마스크 결합: 하나라도 True면 차단
    combined_mask = torch.logical_or(custom_mask, padding_block_mask)
    
    return combined_mask


def convert_mask_to_additive(mask: torch.Tensor, dtype: torch.dtype = torch.float32):
    """
    Boolean mask를 additive mask로 변환 (LLaMA 호환)
    
    LLaMA의 attention 메커니즘은 additive mask를 사용:
    - 0.0: attend 가능
    - -inf: attend 불가
    
    Args:
        mask (torch.Tensor): Boolean mask [batch, 1, seq_len, seq_len]
                            True(1) = blocked, False(0) = allowed
        dtype (torch.dtype): 출력 데이터 타입
        
    Returns:
        torch.Tensor: Additive mask with -inf for blocked positions
    """
    # Boolean mask를 float로 변환
    additive_mask = torch.zeros_like(mask, dtype=dtype)
    additive_mask = additive_mask.masked_fill(mask, float('-inf'))
    
    return additive_mask
