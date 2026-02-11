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


from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config) # 초기화


# LlamaForCausalLM: 기본 LLaMA 모델의 언어 생성요 클래스
# LlavaMetaForCausalLM: 멀티모달 입력(이미지+텍스트)을 처리하는 기능 추가// 이미지가 있으면 LlavaMetaForCausalLM의 prepare_inputs_labels_for_multimodal 사용
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config) # model.model 부분
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _get_adapter_controller(self):
        """
        Return the module that actually owns PEFT adapters.
        In dual-LoRA inference, this is usually the outer PeftModel wrapper.
        """
        candidates = [
            self,
            getattr(self, "_peft_parent_model", None),
            getattr(self, "base_model", None),
            getattr(getattr(self, "base_model", None), "model", None),
            getattr(self, "model", None),
            getattr(getattr(self, "model", None), "_peft_parent_model", None),
        ]

        seen = set()
        for candidate in candidates:
            if candidate is None:
                continue
            obj_id = id(candidate)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if hasattr(candidate, "set_adapter") and hasattr(candidate, "peft_config"):
                return candidate
        return None

    @staticmethod
    def _get_loaded_adapter_names(adapter_controller) -> List[str]:
        if adapter_controller is None:
            return []
        peft_config = getattr(adapter_controller, "peft_config", None)
        if isinstance(peft_config, dict):
            return list(peft_config.keys())
        return []

    @staticmethod
    def _get_active_adapter_name(adapter_controller) -> Optional[str]:
        if adapter_controller is None:
            return None

        active_adapter = getattr(adapter_controller, "active_adapter", None)
        if isinstance(active_adapter, str):
            return active_adapter

        active_adapters_obj = getattr(adapter_controller, "active_adapters", None)
        if active_adapters_obj is not None:
            try:
                active_adapters = active_adapters_obj() if callable(active_adapters_obj) else active_adapters_obj
            except Exception:
                active_adapters = None

            if isinstance(active_adapters, str):
                return active_adapters
            if isinstance(active_adapters, (list, tuple)) and len(active_adapters) > 0:
                return active_adapters[0]

        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

    # inputs_embeds is None: 입력 임베딩이 제공되지 않은 경우
    # 이미지가 제공되면 prepare_inputs_labels_for_multimodal 사용하여 inputs_embeds 생성
    
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        # 상속 받은 LlamaForCausalLM의 forward 메서드 호출
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        use_summary_tokens: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        텍스트 생성 (일반 모드 또는 요약 토큰 모드)
        
        Args:
            inputs: 입력 토큰 ID
            images: 입력 이미지
            image_sizes: 이미지 크기
            use_summary_tokens: True이면 two-stage forward 사용
            **kwargs: 생성 파라미터
        """
        # input_ids가 kwargs에 있으면 inputs로 사용
        if inputs is None and 'input_ids' in kwargs:
            inputs = kwargs.pop('input_ids')
        
        # Summary tokens 모드 체크
        if use_summary_tokens and images is not None:
            return self.generate_with_summary_tokens(
                inputs=inputs,
                images=images,
                image_sizes=image_sizes,
                **kwargs
            )
        
        # 기존 일반 생성 로직
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def generate_with_summary_tokens(
        self,
        inputs: torch.Tensor,
        images: torch.Tensor,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Two-stage forward를 이용한 생성
        
        Stage 1: 이미지 → 요약 토큰 hidden states 추출 (attention mask 적용)
        Stage 2: 요약 hidden states로 답변 생성
        
        Args:
            inputs: 입력 텍스트 토큰 ID [batch_size, seq_len]
            images: 입력 이미지 [batch_size, C, H, W]
            image_sizes: 이미지 크기
            **kwargs: 생성 파라미터 (max_new_tokens, temperature 등)
            
        Returns:
            생성된 토큰 ID
        """
        # ========== Stage 1: 요약 토큰 생성 ==========
        # 고정 프롬프트로 attention mask 적용하여 요약 hidden states 추출
        result = self.prepare_inputs_for_summary_generation_batch(
            images=images,
            image_sizes=image_sizes,
            return_attention_mask=True
        )
        
        if len(result) == 3:
            inputs_embeds_stage1, summary_positions, custom_attention_mask = result
        else:
            inputs_embeds_stage1, summary_positions = result
            custom_attention_mask = None
        
        # LLM forward (hidden states만 추출, 생성 안 함)
        forward_kwargs = {
            'inputs_embeds': inputs_embeds_stage1,
            'output_hidden_states': True,
            'return_dict': True
        }
        
        if custom_attention_mask is not None:
            # 패딩 마스크 생성 (고정 길이라서 모두 유효)
            batch_size, seq_len = inputs_embeds_stage1.shape[0], inputs_embeds_stage1.shape[1]
            padding_mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.bool,
                device=inputs_embeds_stage1.device
            )
            
            # 마스크 결합 및 변환
            from llava.model.attention_utils import combine_masks, convert_mask_to_additive
            combined_mask = combine_masks(custom_attention_mask, padding_mask)
            additive_mask = convert_mask_to_additive(
                combined_mask,
                dtype=inputs_embeds_stage1.dtype
            )
            forward_kwargs['attention_mask'] = additive_mask
        
        # Dual LoRA: Stage 1에서는 'summary_utilizer' adapter 사용
        use_dual_lora = getattr(self.config, 'use_dual_lora', False)
        adapter_controller = self._get_adapter_controller() if use_dual_lora else None
        loaded_adapters = self._get_loaded_adapter_names(adapter_controller)
        original_adapter = self._get_active_adapter_name(adapter_controller)

        if use_dual_lora:
            if adapter_controller is None:
                warnings.warn(
                    "use_dual_lora=True but adapter controller is unavailable. "
                    "Proceeding without adapter switching."
                )
            elif "summary_utilizer" in loaded_adapters:
                adapter_controller.set_adapter("summary_utilizer")
            else:
                warnings.warn(
                    "Dual LoRA requested but 'summary_utilizer' adapter is missing. "
                    f"Loaded adapters: {loaded_adapters}"
                )
        
        # Stage 1 forward
        with torch.no_grad():
            outputs_stage1 = self.model(**forward_kwargs)
        
        # 요약 hidden states 추출
        last_hidden_states = outputs_stage1.hidden_states[-1]
        summary_hidden_states = self.extract_summary_hidden_states(
            last_hidden_states,
            summary_positions
        )  # [batch_size, num_summary_tokens, hidden_size]
        
        # ========== Stage 2: 요약으로 답변 생성 ==========
        # Dual LoRA: Stage 2에서는 'default' adapter 사용
        if use_dual_lora and adapter_controller is not None:
            if "default" in loaded_adapters:
                adapter_controller.set_adapter("default")
            else:
                warnings.warn(
                    "Dual LoRA requested but 'default' adapter is missing. "
                    f"Loaded adapters: {loaded_adapters}"
                )
        
        # 원본 입력 준비
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        # 요약 hidden states를 이미지 대신 사용하여 입력 준비
        (
            _,
            position_ids_stage2,
            attention_mask_stage2,
            _,
            inputs_embeds_stage2,
            _
        ) = self.prepare_inputs_with_summary(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            summary_hidden_states=summary_hidden_states,
            image_sizes=image_sizes
        )
        
        # Stage 2 생성
        # inputs_embeds를 사용할 때는 dummy input_ids가 필요함
        # transformers generate는 input_ids 길이를 기준으로 출력을 구성함
        batch_size, seq_len = inputs_embeds_stage2.shape[:2]
        dummy_input_ids = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=inputs_embeds_stage2.device
        )
        
        try:
            return super().generate(
                input_ids=dummy_input_ids,
                position_ids=position_ids_stage2,
                attention_mask=attention_mask_stage2,
                inputs_embeds=inputs_embeds_stage2,
                **kwargs
            )
        finally:
            # Keep adapter state stable across repeated inference calls.
            if (
                use_dual_lora
                and adapter_controller is not None
                and original_adapter is not None
                and original_adapter in loaded_adapters
            ):
                current_adapter = self._get_active_adapter_name(adapter_controller)
                if current_adapter != original_adapter:
                    adapter_controller.set_adapter(original_adapter)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
