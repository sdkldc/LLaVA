# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SUMMARY_PROMPT
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

# PyTorch 경고 해결을 위한 패치 적용
from llava.train.llama_checkpoint_patch import apply_all_patches
apply_all_patches()


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # 요약 토큰 관련 파라미터
    use_summary_tokens: bool = field(default=False, metadata={"help": "이미지 요약 토큰 사용 여부"})
    num_summary_tokens: int = field(default=8, metadata={"help": "요약 토큰 개수 (1, 4, 8, 16, 32 등)"})
    tune_summary_tokens: bool = field(default=True, metadata={"help": "요약 토큰 학습 여부"})
    mask_summary_tokens: bool = field(default=True, metadata={"help": "요약 토큰 간 attention mask 적용 여부 (True: 참조 불가, False: 참조 가능)"})
    use_dual_lora: bool = field(default=False, metadata={"help": "1st/2nd forward에 별도 LoRA 사용 여부"})
    # K-means 초기화 관련 파라미터
    kmeans_init: bool = field(default=False, metadata={"help": "K-means로 요약 토큰 초기화 여부"})
    kmeans_metric: str = field(default='cosine', metadata={"help": "K-means 거리 메트릭 (cosine, l2, dot)"})
    kmeans_n_iter: int = field(default=3, metadata={"help": "K-means 반복 횟수"})
    kmeans_apply_point: str = field(default='before_projector', metadata={"help": "K-means 적용 시점 (before_projector: 프로젝터 전, after_projector: 프로젝터 후)"})
    kmeans_use_nearest: bool = field(default=True, metadata={"help": "True: centroid와 가장 가까운 실제 토큰 사용, False: centroid 직접 사용"})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of module name suffixes to apply LoRA to "
                "(e.g., 'q_proj,v_proj,k_proj,o_proj'). If unset, targets all "
                "linear layers except multimodal modules."
            )
        },
    )
    dual_lora_stage1_disable_gc: bool = field(
        default=False,
        metadata={
            "help": (
                "Dual LoRA + DeepSpeed 안정화용 옵션. True이면 1st forward에서만 "
                "gradient checkpointing을 임시 비활성화한다(메모리 사용량 증가 가능)."
            )
        },
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def _get_active_adapter_name(model) -> Optional[str]:
    active_adapter = getattr(model, "active_adapter", None)
    if isinstance(active_adapter, str):
        return active_adapter

    active_adapters_obj = getattr(model, "active_adapters", None)
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


def get_peft_state_for_adapter_maybe_zero_3(model, adapter_name, bias):
    """
    DeepSpeed ZeRO-3 환경에서 특정 adapter의 state_dict를 추출합니다.
    반환 키 형식은 PEFT save_pretrained가 기대하는 형식(어댑터 접미사 제거)입니다.
    """
    original_adapter = _get_active_adapter_name(model)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)

    adapter_suffix = f".{adapter_name}."
    adapter_names = []
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        adapter_names = list(peft_config.keys())
    other_adapter_suffixes = [f".{name}." for name in adapter_names if name != adapter_name]

    adapter_params = []
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue

        # Multi-adapter 이름이 노출되는 경우 (.default. / .summary_utilizer.)를 표준 키로 정규화.
        if adapter_suffix in name:
            normalized_name = name.replace(adapter_suffix, ".")
            adapter_params.append((normalized_name, param))
            continue

        # 다른 adapter 이름이 명시된 파라미터는 제외.
        if any(suffix in name for suffix in other_adapter_suffixes):
            continue

        # 활성 어댑터 이름이 노출되지 않는 PEFT 버전도 지원 (bare key).
        if ".lora_" in name:
            adapter_params.append((name, param))

    if hasattr(model, "set_adapter") and original_adapter is not None:
        model.set_adapter(original_adapter)

    if bias == "none":
        to_return = {k: t for k, t in adapter_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in adapter_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {k: t for k, t in adapter_params if "lora_" in k}
    else:
        raise NotImplementedError

    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def _parse_lora_target_modules(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    # 인자 파싱
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    # 모델 인자, 데이터 인자, 학습 인자
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # vision model 존재
    # LLava model loading
    if model_args.vision_tower is not None: 
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else: # LLama-base LLaVA model loading
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else: # vanilla LLaMA model loading
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    
    # Freeze the backbone model
    # LLM 동결
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Add LoRA adapter
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, PeftModel
        import pathlib
        # LoRA 부착 모듈
        target_modules = _parse_lora_target_modules(training_args.lora_target_modules)
        if target_modules is None:
            target_modules = find_all_linear_names(model)
        
        # Dual LoRA: 1st forward와 2nd forward에 별도 LoRA 사용
        if model_args.use_summary_tokens and model_args.use_dual_lora:
            rank0_print("Setting up Dual LoRA adapters for summary tokens...")
            rank0_print(f"LoRA target modules: {target_modules}")
            
            # DeepSpeed ZeRO-3 환경: 모델 구조만 생성, weights는 Trainer가 로드
            # PEFT 기본 생성 어댑터는 이름이 "default"이며, 여기서는 2nd forward용으로 사용한다.
            lora_config_summary = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            
            # 첫 번째 어댑터 생성 (이름: "default" - 2nd forward용)
            model = get_peft_model(model, lora_config_summary)
            rank0_print("Added first adapter: 'default' (for 2nd forward - summary utilization)")
            
            # LoRA #2: 요약 토큰 생성용 (1st forward)
            lora_config_utilizer = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model.add_adapter("summary_utilizer", lora_config_utilizer)
            rank0_print("Added second adapter: 'summary_utilizer' (for 1st forward - summary generation)")
            
            # 초기 활성 어댑터를 summary_utilizer(1st forward용)로 설정
            model.set_adapter("summary_utilizer")

            # Optimizer 생성 시 두 어댑터 파라미터가 모두 포함되도록 보장.
            # (set_adapter는 비활성 어댑터의 requires_grad를 False로 둘 수 있음)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

            rank0_print(f"Active adapters: {list(model.peft_config.keys())}")
            rank0_print(f"Current active adapter: {model.active_adapter}")
            rank0_print("✓ Dual LoRA setup complete!")
            rank0_print("  - 1st Forward (summary generation): 'summary_utilizer' adapter")
            rank0_print("  - 2nd Forward (summary utilization): 'default' adapter")
            
        else:
            # 기존 단일 LoRA
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            rank0_print(f"LoRA target modules: {target_modules}")
            model = get_peft_model(model, lora_config)

    # 토크나이저 로딩
    # 텍스트 -> 토큰 변환기
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # vision_tower 초기화
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        # 비전 인코더 설정
        vision_tower = model.get_vision_tower() # vision encoder
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        # 요약 토큰 초기화
        if model_args.use_summary_tokens:
            rank0_print(f"Initializing summary tokens with {model_args.num_summary_tokens} tokens...")
            model.config.use_summary_tokens = True
            model.config.num_summary_tokens = model_args.num_summary_tokens
            model.config.use_dual_lora = model_args.use_dual_lora  # Dual LoRA 설정 추가
            model.config.mask_summary_tokens = model_args.mask_summary_tokens  # Attention mask 설정 추가
            
            # K-means 설정 추가
            model.config.kmeans_init = model_args.kmeans_init
            model.config.kmeans_metric = model_args.kmeans_metric
            model.config.kmeans_n_iter = model_args.kmeans_n_iter
            model.config.kmeans_apply_point = model_args.kmeans_apply_point
            model.config.kmeans_use_nearest = model_args.kmeans_use_nearest
            
            from llava.model.summary_tokens import build_summary_tokens
            
            # summary_tokens 모듈 생성
            summary_tokens_module = build_summary_tokens(model.config)
            
            # LoRA 적용 여부에 따라 올바른 경로에 설정
            # get_model()을 통해 실제 LlavaLlamaModel에 접근
            target_model = model.get_model()
            target_model.summary_tokens = summary_tokens_module
            
            if model_args.kmeans_init:
                rank0_print(f"🔧 K-means dynamic initialization enabled")
                rank0_print(f"   Metric: {model_args.kmeans_metric}, Iterations: {model_args.kmeans_n_iter}")
                rank0_print(f"   Apply point: {model_args.kmeans_apply_point}")
                rank0_print(f"   Token selection: {'Nearest real token' if model_args.kmeans_use_nearest else 'Centroid'}")
                rank0_print(f"   Summary tokens will be computed per image (not learned)")
                # K-means 모드에서는 요약 토큰을 학습하지 않음
                for p in summary_tokens_module.parameters():
                    p.requires_grad = False
            else:
                # 기존 방식: 학습 가능한 요약 토큰
                if model_args.tune_summary_tokens:
                    for p in summary_tokens_module.parameters():
                        p.requires_grad = True
                    rank0_print("Summary tokens are trainable.")
                else:
                    for p in summary_tokens_module.parameters():
                        p.requires_grad = False
                    rank0_print("Summary tokens are frozen.")
            
            # dtype 설정, 다른 토큰들과 동일
            summary_tokens_module.to(
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                device=training_args.device
            )
            
            # tokenizer 설정 (LoRA 적용 후에도 접근 가능하도록)
            target_model.tokenizer = tokenizer
            
            rank0_print(f"✓ Summary tokens initialized: {model_args.num_summary_tokens} tokens")
        # MM MLP adapter 설정:: 프로젝터
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    # 4bit/8bit 양자화 시, 일부 모듈의 dtype 조정
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    # dataloaders 생성
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # Checkpoint 확인
    checkpoint_steps = []
    for checkpoint in pathlib.Path(training_args.output_dir).glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        suffix = checkpoint.name.split("-")[-1]
        if suffix.isdigit():
            checkpoint_steps.append((int(suffix), checkpoint))

    resume_checkpoint = None
    use_dual_lora = model_args.use_summary_tokens and model_args.use_dual_lora
    if checkpoint_steps:
        checkpoint_steps.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoint_steps[0][1]

        if use_dual_lora:
            # Dual LoRA resume는 두 adapter 파일이 모두 존재하는 checkpoint만 사용한다.
            for _, checkpoint in checkpoint_steps:
                default_adapter_path = checkpoint / "adapter_model.safetensors"
                utilizer_adapter_path = checkpoint / "summary_utilizer" / "adapter_model.safetensors"
                trainer_state_path = checkpoint / "trainer_state.json"
                if (
                    default_adapter_path.is_file()
                    and utilizer_adapter_path.is_file()
                    and trainer_state_path.is_file()
                ):
                    resume_checkpoint = str(checkpoint)
                    break
                rank0_print(
                    f"Skipping incomplete dual LoRA checkpoint: {checkpoint} "
                    f"(requires adapter_model.safetensors + summary_utilizer/adapter_model.safetensors + trainer_state.json)"
                )

            # 완전한 체크포인트가 없으면 기존 동작 유지(최신 체크포인트 사용) + 경고.
            if resume_checkpoint is None:
                resume_checkpoint = str(latest_checkpoint)
                rank0_print(
                    f"Warning: no complete dual LoRA checkpoint found. Falling back to latest checkpoint: {resume_checkpoint}"
                )
        else:
            resume_checkpoint = str(latest_checkpoint)

        rank0_print(f"Found checkpoint: {resume_checkpoint}")

        # Dual LoRA 사용 시, Trainer resume 전에 두 adapter를 미리 로드
        # (2nd adapter는 checkpoint 하위의 별도 폴더에 저장되므로 명시적으로 읽어온다.)
        if use_dual_lora:
            from peft import set_peft_model_state_dict
            import safetensors.torch
            import torch.distributed as dist

            rank0_print("Loading dual LoRA adapter weights from checkpoint...")

            # 1. default adapter weights 로드
            default_adapter_path = os.path.join(resume_checkpoint, "adapter_model.safetensors")
            if os.path.exists(default_adapter_path):
                rank0_print(f"  Loading 'default' adapter from {default_adapter_path}")
                default_state_dict = safetensors.torch.load_file(default_adapter_path)
                set_peft_model_state_dict(model, default_state_dict, adapter_name="default")
            else:
                rank0_print(f"  Warning: missing default adapter file: {default_adapter_path}")

            # 2. summary_utilizer adapter weights 로드
            utilizer_adapter_path = os.path.join(resume_checkpoint, "summary_utilizer", "adapter_model.safetensors")
            if os.path.exists(utilizer_adapter_path):
                rank0_print(f"  Loading 'summary_utilizer' adapter from {utilizer_adapter_path}")
                utilizer_state_dict = safetensors.torch.load_file(utilizer_adapter_path)
                set_peft_model_state_dict(model, utilizer_state_dict, adapter_name="summary_utilizer")
            else:
                rank0_print(f"  Warning: missing summary_utilizer adapter file: {utilizer_adapter_path}")

            rank0_print("✓ Dual LoRA adapter weights loaded successfully")

            # 모든 ranks 동기화
            if dist.is_initialized():
                dist.barrier()

            rank0_print("Dual LoRA resume: adapter weights preloaded.")
    else:
        rank0_print("No checkpoint found, starting training from scratch")
    
    # 트레이너 생성
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # adapter-only resume에서는 첫 저장이 resume source와 동일해지는 경우가 있어
    # Trainer 쪽에서 한 번만 중복 체크포인트를 자동 제거할 수 있도록 소스 경로를 전달한다.
    if (
        resume_checkpoint
        and model_args.use_summary_tokens
        and model_args.use_dual_lora
        and training_args.deepspeed
    ):
        trainer._dual_lora_resume_source_checkpoint = resume_checkpoint

    # 학습 시작
    if resume_checkpoint:
        rank0_print(f"Resuming training from: {resume_checkpoint}")

        use_dual_lora = model_args.use_summary_tokens and model_args.use_dual_lora
        if use_dual_lora and training_args.deepspeed:
            # Dual LoRA + adapter-only resume 경로:
            # DeepSpeed model/optimizer state 복원은 포맷 불일치로 실패할 수 있어
            # trainer_state(global_step)만 복원하도록 고정한다.
            import transformers.trainer as hf_trainer_module

            if not trainer.args.ignore_data_skip:
                trainer.args.ignore_data_skip = True
                training_args.ignore_data_skip = True
                rank0_print("Dual LoRA resume: forcing ignore_data_skip=True for adapter-only fallback.")

            rank0_print("Dual LoRA resume: restoring trainer_state only (skipping DeepSpeed model/optimizer state).")
            original_ds_loader = hf_trainer_module.deepspeed_load_checkpoint
            original_opt_loader = trainer._load_optimizer_and_scheduler
            try:
                hf_trainer_module.deepspeed_load_checkpoint = lambda *args, **kwargs: None
                trainer._load_optimizer_and_scheduler = lambda *args, **kwargs: None
                trainer.train(resume_from_checkpoint=resume_checkpoint)
            finally:
                hf_trainer_module.deepspeed_load_checkpoint = original_ds_loader
                trainer._load_optimizer_and_scheduler = original_opt_loader
        else:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    # KV Cache 활성화 (추론에서)
    model.config.use_cache = True

    if training_args.lora_enable:
        use_dual_lora = getattr(model.config, 'use_dual_lora', False)
        # Save only LoRA adapter weights
        if use_dual_lora:
            # Root adapter_model.safetensors는 항상 'default' 어댑터를 저장.
            state_dict = get_peft_state_for_adapter_maybe_zero_3(
                model, "default", training_args.lora_bias
            )
        else:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
        # lora 외의 학습 가능한 파라미터들도 저장 (예: vision_tower)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )

        summary_utilizer_state_dict = None
        if use_dual_lora:
            # ZeRO-3 GatheredParameters는 collective 동기화가 필요하므로
            # state_dict 추출은 모든 rank에서 실행한다.
            summary_utilizer_state_dict = get_peft_state_for_adapter_maybe_zero_3(
                model,
                "summary_utilizer",
                training_args.lora_bias
            )

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            if use_dual_lora:
                # multi-adapter 환경에서는 root(default)도 수동 저장으로 고정
                from safetensors.torch import save_file
                import json

                if hasattr(model, "set_adapter"):
                    model.set_adapter("default")
                save_file(
                    state_dict,
                    os.path.join(training_args.output_dir, "adapter_model.safetensors")
                )

                default_adapter_config = model.peft_config["default"].to_dict()
                for key, value in default_adapter_config.items():
                    if isinstance(value, set):
                        default_adapter_config[key] = list(value)
                with open(os.path.join(training_args.output_dir, "adapter_config.json"), "w") as f:
                    json.dump(default_adapter_config, f, indent=2)
            else:
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            
            # Dual LoRA: summary_utilizer 어댑터도 별도로 저장
            if use_dual_lora:
                rank0_print("Saving summary_utilizer adapter...")
                from safetensors.torch import save_file
                import json

                summary_utilizer_dir = os.path.join(training_args.output_dir, "summary_utilizer")
                os.makedirs(summary_utilizer_dir, exist_ok=True)

                # safetensors 형식으로 저장
                if len(summary_utilizer_state_dict) > 0:
                    save_file(
                        summary_utilizer_state_dict,
                        os.path.join(summary_utilizer_dir, "adapter_model.safetensors")
                    )

                    # adapter_config.json도 저장
                    adapter_config = model.peft_config["summary_utilizer"].to_dict()
                    # Convert sets to lists for JSON serialization
                    for key, value in adapter_config.items():
                        if isinstance(value, set):
                            adapter_config[key] = list(value)
                    with open(os.path.join(summary_utilizer_dir, "adapter_config.json"), "w") as f:
                        json.dump(adapter_config, f, indent=2)

                    rank0_print(f"✓ Saved summary_utilizer adapter to {summary_utilizer_dir} ({len(summary_utilizer_state_dict)} parameters)")
                else:
                    rank0_print(f"⚠ Warning: summary_utilizer adapter state_dict is empty!")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


# 임포트 시가 아닌, 직접 실핼할 때, 학습 시작
if __name__ == "__main__":
    train()
