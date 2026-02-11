import os
import json
import warnings
import contextlib
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import List, Optional, Tuple, Union


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def _enable_all_lora_gradients(model):
    """set_adapter 후 비활성 어댑터의 requires_grad도 True로 재설정.
    PEFT의 set_adapter는 비활성 어댑터 파라미터의 requires_grad를 False로 설정하므로,
    end-to-end 학습을 위해 모든 LoRA 파라미터의 gradient를 활성화한다."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def _get_unwrapped_model(model):
    """DDP/DeepSpeed/PEFT 래퍼를 벗겨서 LlavaLlamaForCausalLM 반환."""
    unwrapped = model.module if hasattr(model, 'module') else model
    unwrapped = getattr(unwrapped, 'base_model', unwrapped)
    unwrapped = getattr(unwrapped, 'model', unwrapped)
    return unwrapped


def _get_peft_model(model):
    """DDP/DeepSpeed만 벗겨서 PeftModel 반환 (adapter 제어용)."""
    return model.module if hasattr(model, 'module') else model


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def create_accelerator_and_postprocess(self):
        """
        transformers==4.37 + accelerate==0.21 조합에서 DeepSpeed ZeRO-2/3 + grad accumulation 시
        accelerate.accumulate()가 no_sync()를 호출하며 DeepSpeed assert가 발생한다.
        DeepSpeed 엔진은 자체 accumulation을 처리하므로, 해당 경로에서는 no_sync를 no-op으로 바꾼다.
        """
        super().create_accelerator_and_postprocess()

        if not getattr(self, "is_deepspeed_enabled", False):
            return

        accelerator = self.accelerator
        if getattr(accelerator, "_llava_no_sync_patched_for_deepspeed", False):
            return

        original_no_sync = accelerator.no_sync

        @contextlib.contextmanager
        def _deepspeed_safe_no_sync(model):
            # DeepSpeed ZeRO gradient partitioning(no_sync 비호환) 경로는 동기화를 유지.
            if hasattr(model, "zero_optimization_partition_gradients"):
                yield
                return

            with original_no_sync(model):
                yield

        accelerator.no_sync = _deepspeed_safe_no_sync
        accelerator._llava_no_sync_patched_for_deepspeed = True


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def training_step(self, model, inputs):
        """
        Training step with aggressive memory management for Dual LoRA.
        """
        # 기본 training step 실행
        loss = super().training_step(model, inputs)
        
        # Dual LoRA 사용 시: gradient accumulation cycle 완료 후에만 캐시 정리
        # (중간에 정리하면 gradient 계산에 영향)
        use_dual_lora = getattr(getattr(model, 'config', None), 'use_dual_lora', False)
        if use_dual_lora and torch.cuda.is_available():
            # Gradient accumulation의 마지막 스텝에서만 실행
            if self.accelerator.sync_gradients:
                torch.cuda.empty_cache()
        
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Two-stage forward로 dual LoRA end-to-end 학습.

        Stage 1 (summary_utilizer LoRA):
          [고정프롬프트] + [이미지토큰] + [대표토큰] → LLM → summary hidden states
        Stage 2 (default LoRA):
          [텍스트 프롬프트] + [summary hidden states] → LLM → loss

        Gradient checkpointing 호환:
          - GC를 유지한 채 양 stage 모두 실행
          - summary_hidden_states에 backward hook을 등록하여,
            Stage 2 backward 완료 후 Stage 1 backward 시작 전에
            adapter를 "summary_utilizer"로 자동 전환
          - 이로써 GC recomputation 시 올바른 adapter 사용 보장

        이미지가 없거나 dual LoRA가 비활성이면 기존 단일 forward 사용.
        """
        # --- dual LoRA two-stage 학습 조건 확인 ---
        base_model = _get_unwrapped_model(model)
        config = getattr(base_model, 'config', None)
        use_dual_lora = getattr(config, 'use_dual_lora', False)
        use_summary_tokens = getattr(config, 'use_summary_tokens', False)
        images = inputs.get('images', None)

        if not (use_dual_lora and use_summary_tokens and images is not None):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # --- 입력 분리 ---
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs.get('attention_mask', None)
        image_sizes = inputs.get('image_sizes', None)

        # --- adapter controller 확보 ---
        peft_model = _get_peft_model(model)
        adapter_controller = None
        for candidate in [peft_model, getattr(peft_model, 'base_model', None)]:
            if candidate is not None and hasattr(candidate, 'peft_config') and hasattr(candidate, 'set_adapter'):
                adapter_controller = candidate
                break

        if adapter_controller is None:
            warnings.warn(
                "Dual LoRA 학습이 설정되었지만 adapter controller를 찾을 수 없습니다. "
                "단일 forward로 fallback합니다."
            )
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # --- tokenizer 설정 ---
        if getattr(base_model, 'tokenizer', None) is None:
            base_model.tokenizer = self.tokenizer

        # ================================================================
        # Stage 1: Summary Generation (summary_utilizer adapter)
        # GC 유지 - backward hook으로 adapter 전환 보장
        # ================================================================
        adapter_controller.set_adapter("summary_utilizer")
        _enable_all_lora_gradients(model)

        # Stage 1 입력 준비: [고정프롬프트] + [이미지토큰] + [대표토큰]
        result = base_model.prepare_inputs_for_summary_generation_batch(
            images=images,
            image_sizes=image_sizes,
            return_attention_mask=True
        )
        inputs_embeds_s1, summary_positions, custom_attention_mask = result

        # attention mask 처리
        forward_kwargs_s1 = {
            'inputs_embeds': inputs_embeds_s1,
            'return_dict': True,
        }

        if custom_attention_mask is not None:
            from llava.model.attention_utils import convert_mask_to_additive

            # padding_mask 불필요 (모든 토큰이 유효하므로 custom_mask만 사용)
            additive_mask = convert_mask_to_additive(custom_attention_mask, dtype=inputs_embeds_s1.dtype)
            forward_kwargs_s1['attention_mask'] = additive_mask

        # Stage 1 LLM forward (labels/logits 계산 없이 backbone hidden state만 추출)
        # 기존 model(...) 경로는 logits + (optionally) all hidden states를 생성해
        # 메모리 피크를 키우므로, 동일한 마지막 hidden state만 사용하는 경로로 제한한다.
        forward_kwargs_s1['use_cache'] = False
        llm_backbone = base_model.get_model() if hasattr(base_model, "get_model") else None
        if llm_backbone is None:
            warnings.warn(
                "Base model backbone을 찾을 수 없어 Stage 1에서 full model forward로 fallback합니다."
            )
            forward_kwargs_s1['output_hidden_states'] = True
            outputs_s1 = model(**forward_kwargs_s1)
            last_hidden_states = outputs_s1.hidden_states[-1]
        else:
            outputs_s1 = llm_backbone(**forward_kwargs_s1)
            last_hidden_states = outputs_s1.last_hidden_state

        # view가 원본 전체 hidden state storage를 잡고 있지 않도록 contiguous tensor로 분리
        summary_hidden_states = base_model.extract_summary_hidden_states(
            last_hidden_states, summary_positions
        ).contiguous()  # [batch_size, num_summary_tokens, hidden_size]
        
        # Stage 1 중간 텐서 명시적 해제 (메모리 압박 완화)
        del outputs_s1
        del last_hidden_states
        del inputs_embeds_s1
        if custom_attention_mask is not None:
            del custom_attention_mask
        
        # Dual LoRA에서 Stage 1 후 메모리 정리 (메모리 압박 시 필수)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # Backward hook: Stage 2 backward 완료 → adapter를 summary_utilizer로 전환
        # Stage 1의 GC recomputation이 올바른 adapter를 사용하도록 보장
        # 
        # 메모리 누수 방지: hook handle을 저장하여 backward 후 제거
        # ================================================================
        hook_handle = None
        
        def _switch_adapter_for_stage1_backward(grad):
            adapter_controller.set_adapter("summary_utilizer")
            _enable_all_lora_gradients(model)
            # Hook 실행 후 자동 제거 (메모리 누수 방지)
            if hook_handle is not None:
                hook_handle.remove()
            return grad

        hook_handle = summary_hidden_states.register_hook(_switch_adapter_for_stage1_backward)

        # ================================================================
        # Stage 2: Main Forward with Loss (default adapter)
        # ================================================================
        adapter_controller.set_adapter("default")
        _enable_all_lora_gradients(model)

        # Stage 2 입력 준비: 이미지 토큰 위치에 summary hidden states 삽입
        (
            _,
            position_ids_s2,
            attention_mask_s2,
            _,
            inputs_embeds_s2,
            labels_s2
        ) = base_model.prepare_inputs_with_summary(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            summary_hidden_states=summary_hidden_states,
            image_sizes=image_sizes
        )
        
        # summary_hidden_states는 embeds에 복사되었으므로 이제 필요 없음
        # 단, backward hook이 실행될 때까지 computational graph에는 남아있음
        # Python reference만 해제 (메모리 누수 방지)
        del summary_hidden_states

        # Stage 2 LLM forward (labels 포함 → loss 계산)
        forward_kwargs_s2 = {
            'inputs_embeds': inputs_embeds_s2,
            'labels': labels_s2,
            'return_dict': True,
            'use_cache': False,
        }
        if attention_mask_s2 is not None:
            forward_kwargs_s2['attention_mask'] = attention_mask_s2
        if position_ids_s2 is not None:
            forward_kwargs_s2['position_ids'] = position_ids_s2

        outputs_s2 = model(**forward_kwargs_s2)
        loss = outputs_s2.loss

        # Stage 2 중간 텐서 정리 (return_outputs=False 시)
        if not return_outputs:
            # outputs_s2의 logits 등 불필요한 텐서 해제
            del outputs_s2
            # Stage 2 inputs도 명시적 해제
            del inputs_embeds_s2
            del labels_s2
            if attention_mask_s2 is not None:
                del attention_mask_s2
            if position_ids_s2 is not None:
                del position_ids_s2
            
            # summary_hidden_states도 backward 후 필요 없으므로 해제
            # (backward hook이 실행된 후에는 더 이상 필요 없음)
            # 단, backward 전이므로 computational graph에는 여전히 연결됨
        
        return (loss, outputs_s2) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            
            # weight_to_save 즉시 해제
            del weight_to_save
        else:
            # DeepSpeed ZeRO-3 checkpoint 저장 전 메모리 확보
            if self.is_deepspeed_enabled and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

        # ZeRO-3 checkpoint 저장 직후 적극적인 메모리 정리
        if torch.cuda.is_available():
            # GPU 동기화 후 캐시 정리
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # DeepSpeed가 활성화된 경우 추가 메모리 정리
            if self.is_deepspeed_enabled:
                # 모든 GPU 동기화
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                
                # 추가 캐시 정리
                torch.cuda.empty_cache()
                
                # allocator 통계 리셋으로 fragmentation 완화
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
