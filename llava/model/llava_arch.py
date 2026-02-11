#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .summary_tokens import build_summary_tokens

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, SUMMARY_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SUMMARY_PROMPT

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
        
        # 요약 토큰 모듈 초기화
        # summary_tokens.py 의 build_summary_tokens 사용
        if getattr(config, 'use_summary_tokens', False):
            self.summary_tokens = build_summary_tokens(config)
            # 모델의 dtype과 맞추기 (BFloat16 등)
            if hasattr(self, 'dtype') and self.dtype is not None:
                self.summary_tokens = self.summary_tokens.to(dtype=self.dtype)
        else:
            self.summary_tokens = None

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    # 46-49 line 정의된 summary_tokens 모듈 반환
    def get_summary_tokens(self):
        """요약 토큰 모듈 반환 - LoRA 래퍼 고려
        
        get_model()을 통해 얻은 모델에서 summary_tokens를 찾습니다.
        이미 올바른 모델 인스턴스(LlavaLlamaModel)에 접근한 상태이므로
        직접 summary_tokens를 반환합니다.
        """
        model = self.get_model()
        
        # get_model()은 이미 LoRA 래퍼를 벗긴 실제 모델을 반환하므로
        # 직접 접근 시도
        summary_tokens = getattr(model, 'summary_tokens', None)
        return summary_tokens
    
    # 이미지->비전 인코더->프로젝터:: 이미지 특징 추출
    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        mm_projector = self.get_model().mm_projector
        
        # vision_tower의 dtype에 맞춰 이미지 변환
        vt_dtype = next(vision_tower.parameters()).dtype
        images = images.to(dtype=vt_dtype)
        
        image_features = vision_tower(images)
        
        # mm_projector의 dtype에 맞춰 image_features 변환
        mm_projector_dtype = next(mm_projector.parameters()).dtype
        image_features = image_features.to(dtype=mm_projector_dtype)
        
        image_features = mm_projector(image_features)
        
        return image_features
    
    # loss 계산에서 호출
    # 입력: 이미지, 이미지 사이즈 -> inputs_embeds, 요약 토큰 위치, attention_mask 반환
    def prepare_inputs_for_summary_generation_batch(
        self, images, image_sizes=None, return_attention_mask=True
    ):
        """
        배치 처리를 위한 요약 토큰 생성 입력 준비
        "Summarize the image in a few words." + 이미지 + 요약 토큰으로 구성
        
        Args:
            images: 입력 이미지 배치 [batch_size, C, H, W]
            image_sizes: 이미지 크기 정보
            return_attention_mask: 커스텀 attention mask 반환 여부
            
        Returns:
            inputs_embeds: 배치 텍스트 + 이미지 + 요약 토큰 임베딩 [batch_size, seq_len, hidden_size]
            summary_token_positions: 요약 토큰의 위치 정보 (start, end) 튜플
            attention_mask: 요약 토큰 간 참조 방지 마스크 [batch_size, 1, seq_len, seq_len] (옵션)
        """
        vision_tower = self.get_vision_tower()
        summary_tokens_module = self.get_summary_tokens() # summary_tokens.py 의 SummaryTokens 모듈
        
        if vision_tower is None:
            raise ValueError(f"Vision tower is not available")
        if images is None:
            raise ValueError(f"Images is None")
        if summary_tokens_module is None:
            raise ValueError(f"Summary tokens module is not available")
        
        # images 차원 검증 및 변환
        if isinstance(images, list):
            images = torch.stack([img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images])
        
        if images.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]

        # K-means before_projector 모드: 비전 타워 1회 호출 후 raw features 재사용
        # (기존: encode_images + vision_tower 이중 호출 → 수정: vision_tower 1회)
        _kmeans_cfg = getattr(self.config, 'kmeans_init', False)
        _kmeans_ap = getattr(self.config, 'kmeans_apply_point', 'before_projector')
        _need_raw = _kmeans_cfg and _kmeans_ap == 'before_projector'
        _vision_raw_for_kmeans = None

        if _need_raw:
            # 비전 타워 1회 호출 → raw features → projector 분리 실행
            _vt = self.get_model().get_vision_tower()
            _proj = self.get_model().mm_projector
            _vt_dtype = next(_vt.parameters()).dtype
            _proj_dtype = next(_proj.parameters()).dtype
            _vision_raw = _vt(images.to(dtype=_vt_dtype))
            if isinstance(_vision_raw, (list, tuple)):
                _vision_raw = _vision_raw[0]
            image_features = _proj(_vision_raw.to(dtype=_proj_dtype))
            # kmeans용 raw features (detach로 gradient 분리)
            _vision_raw_for_kmeans = _vision_raw.detach()
            del _vision_raw
        else:
            # 표준 경로: encode_images (vision tower + projector)
            image_features = self.encode_images(images)

        if isinstance(image_features, list):
            image_features = torch.stack(image_features, dim=0)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        # 고정 프롬프트 토큰화: "Summarize the image in a few words."
        # LoRA wrapped 모델에서도 동작하도록 여러 경로 시도
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            # LoRA 적용 시 base_model.model.tokenizer에 있을 수 있음
            tokenizer = getattr(getattr(self, 'base_model', self), 'tokenizer', None)
        if tokenizer is None:
            # 또는 model.tokenizer에 있을 수 있음
            tokenizer = getattr(self.get_model(), 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer is not available. Set model.tokenizer first.")
        
        prompt_ids = tokenizer.encode(SUMMARY_PROMPT, add_special_tokens=False, return_tensors='pt')
        prompt_ids = prompt_ids.to(self.device)
        
        # 프롬프트 임베딩 (배치 확장), 동일한 프롬프트를 각 배치에 동일하게 사용
        prompt_embeds = self.get_model().embed_tokens(prompt_ids)  # [1, prompt_len, hidden_size]
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)  # [batch_size, prompt_len, hidden_size]
        # dtype 맞추기: image_features와 동일한 dtype 사용
        prompt_embeds = prompt_embeds.to(dtype=image_features.dtype)
        
        # 요약 토큰 임베딩 생성
        # K-means 초기화 활성화 시: 매 forward마다 이미지에서 대표 토큰 추출
        # K-means 비활성화 시: 학습 가능한 요약 토큰 사용
        kmeans_config = getattr(self.config, 'kmeans_init', False)
        
        if kmeans_config:
            # 동적 K-means: 매번 이미지에서 대표 토큰 계산
            from llava.model.kmeans_initializer import initialize_summary_tokens_with_kmeans, kmeans_clustering, find_nearest_tokens_to_centroids
            
            kmeans_metric = getattr(self.config, 'kmeans_metric', 'cosine')
            kmeans_n_iter = getattr(self.config, 'kmeans_n_iter', 3)
            kmeans_apply_point = getattr(self.config, 'kmeans_apply_point', 'before_projector')
            kmeans_use_nearest = getattr(self.config, 'kmeans_use_nearest', True)
            num_summary_tokens = getattr(self.config, 'num_summary_tokens', 8)
            
            with torch.no_grad():
                mm_projector = self.get_model().mm_projector

                if kmeans_apply_point == 'before_projector':
                    # 기본 방식: Vision Encoder → K-means → Centroid/Nearest → Projector
                    # 위에서 비전 타워 1회 호출 시 저장된 raw features 재사용 (이중 호출 방지)
                    projector_dtype = next(mm_projector.parameters()).dtype
                    vision_features_raw = _vision_raw_for_kmeans.to(dtype=projector_dtype)

                    summary_embeds_list = []

                    for i in range(batch_size):
                        single_vision_features = vision_features_raw[i] if vision_features_raw.dim() == 3 else vision_features_raw

                        summary_tokens_for_image = initialize_summary_tokens_with_kmeans(
                            vision_features=single_vision_features,
                            mm_projector=mm_projector,
                            num_summary_tokens=num_summary_tokens,
                            metric=kmeans_metric,
                            n_iter=kmeans_n_iter,
                            random_state=42 + i,
                            use_nearest=kmeans_use_nearest
                        )  # [num_summary_tokens, hidden_size]
                        summary_embeds_list.append(summary_tokens_for_image)

                    del vision_features_raw  # kmeans 완료 후 해제
                
                elif kmeans_apply_point == 'after_projector':
                    # 대안 방식: Vision Encoder → Projector → K-means → Centroid/Nearest
                    # 이미 프로젝터를 통과한 image_features 사용
                    # dtype 확인
                    projector_dtype = next(mm_projector.parameters()).dtype
                    image_features = image_features.to(dtype=projector_dtype)
                    
                    summary_embeds_list = []
                    
                    for i in range(batch_size):
                        # 단일 이미지의 projected features
                        single_image_features = image_features[i] if image_features.dim() == 3 else image_features
                        
                        # K-means는 학습되지 않으므로 no_grad로 감싸서 경고 방지
                        with torch.no_grad():
                            # 프로젝터 후 feature에서 K-means
                            centroids = kmeans_clustering(
                                embeddings=single_image_features,
                                num_clusters=num_summary_tokens,
                                metric=kmeans_metric,
                                n_iter=kmeans_n_iter,
                                random_state=42 + i
                            )  # [num_summary_tokens, hidden_size]
                            
                            # 토큰 선택
                            if kmeans_use_nearest:
                                # Centroid와 가장 가까운 실제 projected 토큰 사용
                                representative_tokens = find_nearest_tokens_to_centroids(
                                    embeddings=single_image_features,
                                    centroids=centroids,
                                    metric=kmeans_metric
                                )
                            else:
                                # Centroid 직접 사용
                                representative_tokens = centroids
                        
                        summary_embeds_list.append(representative_tokens)
                
                else:
                    raise ValueError(f"Invalid kmeans_apply_point: {kmeans_apply_point}. Use 'before_projector' or 'after_projector'.")
                
                # 배치로 스택
                summary_embeds = torch.stack(summary_embeds_list, dim=0)  # [batch_size, num_summary_tokens, hidden_size]
        else:
            # 기존 방식: 학습 가능한 요약 토큰 사용
            summary_embeds = summary_tokens_module.forward(batch_size)  # [batch_size, num_summary_tokens, hidden_size]
        
        # 모든 임베딩을 image_features.dtype으로 통일
        target_dtype = image_features.dtype
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
        summary_embeds = summary_embeds.to(dtype=target_dtype, device=image_features.device)
        
        # 임베딩 결합: [프롬프트] + [이미지] + [요약 토큰]
        inputs_embeds = torch.cat([
            prompt_embeds,      # [batch_size, prompt_len, hidden_size]
            image_features,     # [batch_size, num_image_tokens, hidden_size]
            summary_embeds      # [batch_size, num_summary_tokens, hidden_size]
        ], dim=1)
        
        # 요약 토큰 위치 계산 (모든 샘플에서 동일)
        prompt_len = prompt_embeds.shape[1]
        image_len = image_features.shape[1]
        summary_start = prompt_len + image_len
        summary_end = summary_start + summary_embeds.shape[1]
        summary_token_positions = (summary_start, summary_end)
        
        # 커스텀 Attention Mask 생성 (요약 토큰 간 참조 방지)
        attention_mask = None
        if return_attention_mask:
            # mask_summary_tokens 설정 확인 (기본값 True)
            mask_summary_tokens = getattr(self.config, 'mask_summary_tokens', True)
            
            if mask_summary_tokens:
                # 요약 토큰 간 참조를 방지하는 attention mask 생성
                from llava.model.attention_utils import create_summary_token_attention_mask_optimized
                
                seq_length = inputs_embeds.shape[1] # 전체 시퀀스 길이
                attention_mask = create_summary_token_attention_mask_optimized(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    summary_token_positions=summary_token_positions,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )  # [batch_size, 1, seq_len, seq_len], Boolean mask (True=blocked)
            else:
                # mask_summary_tokens=False: 요약 토큰 간 참조 가능, attention_mask 미생성
                attention_mask = None
        
        # inputs_embeds: 고정 프롬프트 + 이미지 + 요약 토큰 임베딩
        # summary_token_positions: (start_idx, end_idx) 튜플, 요약 토큰 위치
        # attention_mask: 요약 토큰 간 참조 방지를 위한 커스텀 마스크 (옵션)
        if return_attention_mask:
            return inputs_embeds, summary_token_positions, attention_mask
        else:
            return inputs_embeds, summary_token_positions 
    
    
    # 사용 x 
    def prepare_inputs_for_summary_generation(
        self, input_ids, attention_mask, images, image_sizes=None
    ):
        """
        단일 샘플 요약 토큰 생성을 위한 입력 준비 (하위 호환성)
        "Summarize the image in a few words." + 이미지 + 요약 토큰으로 구성
        
        Args:
            input_ids: 원본 텍스트 토큰 ID (사용하지 않고 고정 프롬프트 사용)
            attention_mask: Attention mask
            images: 입력 이미지
            image_sizes: 이미지 크기 정보
            
        Returns:
            inputs_embeds: 텍스트 + 이미지 + 요약 토큰 임베딩
            summary_token_positions: 요약 토큰의 위치 정보 (hidden states 추출용)
        """
        vision_tower = self.get_vision_tower()
        summary_tokens_module = self.get_summary_tokens()
        
        if vision_tower is None or images is None or summary_tokens_module is None:
            raise ValueError("Vision tower, images, or summary tokens module is not available")
        
        # 이미지 인코딩
        image_features = self.encode_images(images)
        if isinstance(image_features, list):
            image_features = image_features[0]  # 단일 이미지 처리
        
        # 고정 프롬프트 토큰화
        # "Summarize the image in a few words."
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer is not available. Set model.tokenizer first.")
        
        prompt_ids = tokenizer.encode(SUMMARY_PROMPT, add_special_tokens=False, return_tensors='pt')
        prompt_ids = prompt_ids.to(self.device)
        
        # 프롬프트 임베딩
        prompt_embeds = self.get_model().embed_tokens(prompt_ids)  # [1, prompt_len, hidden_size]
        
        # 요약 토큰 임베딩
        summary_embeds = summary_tokens_module.get_single_batch()  # [num_summary_tokens, hidden_size]
        summary_embeds = summary_embeds.unsqueeze(0)  # [1, num_summary_tokens, hidden_size]
        
        # 임베딩 결합: [프롬프트] + [이미지] + [요약 토큰]
        # image_features shape: [num_image_tokens, hidden_size] or [1, num_image_tokens, hidden_size]
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        inputs_embeds = torch.cat([
            prompt_embeds,           # [1, prompt_len, hidden_size]
            image_features,          # [1, num_image_tokens, hidden_size]
            summary_embeds           # [1, num_summary_tokens, hidden_size]
        ], dim=1)
        
        # 요약 토큰 위치 계산 (모든 샘플에서 동일)
        prompt_len = prompt_embeds.shape[1]
        image_len = image_features.shape[1]
        summary_start = prompt_len + image_len
        summary_end = summary_start + summary_embeds.shape[1]
        summary_token_positions = (summary_start, summary_end)
        
        return inputs_embeds, summary_token_positions
    
    def extract_summary_hidden_states(self, hidden_states, summary_token_positions):
        """
        LLM의 마지막 레이어에서 요약 토큰의 hidden states 추출
        
        Args:
            hidden_states: LLM의 출력 hidden states [batch_size, seq_len, hidden_size]
            summary_token_positions: (start_idx, end_idx) 튜플
            
        Returns:
            summary_hidden_states: 요약 토큰의 hidden states [batch_size, num_summary_tokens, hidden_size]
        """
        start_idx, end_idx = summary_token_positions
        summary_hidden_states = hidden_states[:, start_idx:end_idx, :]
        return summary_hidden_states
    
    def prepare_inputs_with_summary(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        summary_hidden_states, image_sizes=None
    ):
        """
        2nd Forward: 요약 hidden states를 사용하여 입력 준비
        이미지 대신 요약 정보를 사용하여 텍스트 + 요약으로 forward
        
        LLM 입력 전처리: 패딩, 마스킹 처리, 요약 토큰 삽입
        
        Args:
            input_ids: 텍스트 토큰 ID [batch_size, seq_len]
            position_ids: Position IDs
            attention_mask: Attention mask
            past_key_values: Past key values
            labels: 학습 레이블
            summary_hidden_states: 1st forward에서 생성된 요약 [batch_size, num_summary_tokens, hidden_size]
            image_sizes: 이미지 크기 (사용 안 함)
            
        Returns:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
        """
        # 기존 prepare_inputs_labels_for_multimodal과 유사하지만
        # 이미지 토큰 위치에 summary_hidden_states를 삽입
        
        if input_ids.shape[1] == 1:  # 생성 중일 때는 요약 사용 안 함
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # 더미 처리
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # 패딩 제거
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # 텍스트 임베딩과 요약 hidden states 결합
        new_input_embeds = []
        new_labels = []
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # IMAGE_TOKEN_INDEX 위치 찾기
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            if num_images == 0:
                # 이미지 토큰이 없으면 텍스트만 사용
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # 이미지 토큰 위치 찾기
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim = []
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(labels[batch_idx][image_token_indices[i]+1:image_token_indices[i+1]])
            
            # 텍스트 임베딩
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 텍스트 + 요약 결합
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_summary_idx = 0
            
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                
                if i < num_images:
                    # 이미지 토큰 대신 요약 hidden states 삽입
                    cur_summary = summary_hidden_states[batch_idx]  # [num_summary_tokens, hidden_size]
                    cur_new_input_embeds.append(cur_summary)
                    cur_new_labels.append(
                        torch.full((cur_summary.shape[0],), IGNORE_INDEX, 
                                   device=labels[batch_idx].device, dtype=labels[batch_idx].dtype)
                    )
                    cur_summary_idx += 1
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Padding (메모리 최적화: zero 텐서 재사용)
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        # 첫 번째 샘플의 hidden_size 가져오기
        hidden_size = new_input_embeds[0].shape[1]
        dtype = new_input_embeds[0].dtype
        device = new_input_embeds[0].device
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, 
                                       dtype=new_labels[0].dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), 
                                     dtype=attention_mask.dtype, device=device)
        position_ids = torch.zeros((batch_size, max_len), 
                                   dtype=position_ids.dtype, device=device)
        
        # padding side 확인
        pad_left = getattr(self.config, 'tokenizer_padding_side', 'right') == "left"
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            pad_len = max_len - cur_len
            
            if pad_len > 0:
                # zero padding 생성 (한 번만)
                padding = torch.zeros((pad_len, hidden_size), dtype=dtype, device=device)
                
                if pad_left:
                    new_input_embeds_padded.append(torch.cat((padding, cur_new_embed), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
                else:
                    new_input_embeds_padded.append(torch.cat((cur_new_embed, padding), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
            else:
                # padding 불필요
                new_input_embeds_padded.append(cur_new_embed)
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    # 이미지 -> vision_tower -> projector -> inputs_embeds 생성
    # 택스트 입력과 이미지 입력을 결합하여 inputs_embeds 생성
    # 시퀀스가 샘플마다 달라지니, 패딩 처리를 통한 병렬처리
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1: # 이미지 없음
            return input_ids, position_ids, attention_mask, past_key_values, None, labels # inputs_embeds 없음

        if type(images) is list or images.ndim == 5: # 이미지가 여러 개이거나 5차원 텐서인 경우
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] # 단일 이미지에 대해서 배치 차원 추가:: 만약 [C,H,W] 형태면 [1,C,H,W]로 변환
            concat_images = torch.cat([image for image in images], dim=0) # 모든 이미지 배치로 결합:: shape:[total_B,C,H,W]
            image_features = self.encode_images(concat_images) # 이미지 특징 추출, 인코더+프로젝터
            # [... for ... in ...]:: (ex) [x*2 for x in [1,2,3]] -> [2,4,6]
            split_sizes = [image.shape[0] for image in images] # image.shape[0]:: 각 이미지 샘플 크기 리스트:: 하나의 입력에 여러개의 이미지 입력 
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images) # 이미지 특징 추출, 인코더+프로젝터

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # _ 가공전 원본
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        # attention_mask 가 None 인 경우 모두 1로 채워진 텐서 생성
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            # bool 타입으로 변환
            attention_mask = attention_mask.bool()
        if position_ids is None:
            # 없으면 0,...,len-1 까지의 텐서 생성
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        if getattr(self.config, 'mm_use_im_start_end', False):
            im_start_token_id = getattr(self.config, 'im_start_token_id', None)
            im_end_token_id = getattr(self.config, 'im_end_token_id', None)
            if im_start_token_id is not None and im_end_token_id is not None:
                special_mask = (input_ids == im_start_token_id) | (input_ids == im_end_token_id)
                labels.masked_fill_(special_mask, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # 토큰의 패딩 제거, attention_mask가 True인 위치만 남김
        # input_ids: [batch_size, seq_len], 토큰 ID
        # attention_mask: [batch_size, seq_len], 패딩 위치 False
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # text embedding과 image embedding 결합
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # 여기서 사용되는 이미지 토큰 인덱스는 실제 이미지 인코더를 통과하고 나온 토큰이 아닌, 자리표시용 토큰임에 유의
        for batch_idx, cur_input_ids in enumerate(input_ids): # cur_input_ids: 토큰 1D 텐서
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # num_images: 이미지 토큰 수
            # 이미지 토큰 없는 샘플의 경우
            # 빈 이미지 특징 텐서 생성 
            # 형식상 이미지 인덱스 하나 차례로 사용
            if num_images == 0: 
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids) # 텍스트 임베딩
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0) # 이미지 특징은 빈 텐서로 결합
                new_input_embeds.append(cur_input_embeds) # 최종 입력 임베딩 리스트에 추가
                new_labels.append(labels[batch_idx])    # 최종 레이블 리스트에 추가
                cur_image_idx += 1
                continue 
            # cur_input_ids.shape: 토큰 개수  
            # 이미지 토큰 인덱스 찾음 
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # 이미지 토큰 인덱스 사이의 텍스트 토큰 추출, len(image_token_indices): 이미지 토큰 개수 + 2 
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # 이미지 토큰 제외한 텍스트 토큰
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]]) # 이미지 토큰 제외한 레이블
            split_sizes = [x.shape[0] for x in cur_labels_noim] # 텍스트 토큰 길이 리스트
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # 텍스트 임베딩으로 변환
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # 텍스트 임베딩 분할
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i]) # 이미지 토큰 전 텍스트 임베딩
                cur_new_labels.append(cur_labels_noim[i]) # 이미지 토큰 전 레이블
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features) # 이미지 임베딩 추가
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) # 이미지 임베딩에 해당하는 레이블은 무시하도록 설정
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # 텍스트+이미지 임베딩 시퀀스 길이가 너무 길면 자르기
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them 
        # new_input_embeds[b]: b번째 샘플의 텍스트+이미지 임베딩 2D 텐서 [L_b, hidden_size]
        max_len = max(x.shape[0] for x in new_input_embeds) # 배치 내에서 가장 긴 시퀀스 길이
        batch_size = len(new_input_embeds) # 배치 크기

        # 패딩하는 이유는 각 배치 샘플의 시퀀스 길이가 달라, 패딩을 통해 길이를 맞추고 GPU 병렬 처리
        new_input_embeds_padded = [] # 패딩을 추가한 최종 입력 임베딩 리스트
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        # 패딩 참조하지 않기 위한 마스크
        # 나중에 이곳에 요약 토큰을 보지 못하게 하는 attention mask 생성에 이용할 수 있겠다.
        # 아 안될듯. 왜냐면 서로 참조 같은 경우는 LLM의 attention mask가 담당하고 LLM 부분을 손봐야한다.
        # LLM에 입력되는 것 같다.
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # cur_new_embed: b번째 샘플의 텍스트+이미지 임베딩 2D 텐서 [L_b, hidden_size]
        # cur_new_labels: b번째 샘플의 레이블 1D 텐서 [L_b]
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0] # 토큰 길이
            # 패딩 방향 설정에 따라 앞이나 뒤에 패딩 추가
            # 대부분은 right 인데, 일부 left인 경우도 있어서 처리
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # max_len - cur_len 만큼 뒤에 패딩 추가, hidden_size 차원은 유지
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    # 앞쪽 cur_len 위치에 실제 레이블 복사, 나머지는 ignore_index에 의해 무시
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    # 실제 토큰위치만 True 설정하여 토큰이 아닌 부분은 attention에서 무시
                    attention_mask[i, :cur_len] = True
                    # 실제 토큰 위치에만 포지션 부여
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.config.im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            self.config.im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
