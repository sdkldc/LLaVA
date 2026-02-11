#!/usr/bin/env python
"""
간단한 토큰 전달 테스트 스크립트
단일 샘플로 forward pass를 실행하고 각 단계의 shape을 확인합니다.
"""

import os
import sys
import argparse

# GPU 설정을 가장 먼저 (import torch 전에)
def parse_args_early():
    """GPU 설정을 위해 args를 먼저 파싱"""
    parser = argparse.ArgumentParser(description="토큰 전달 과정 간단 테스트")
    parser.add_argument("--gpu", type=int, default=1, help="사용할 GPU 번호 (기본값: 1)")
    args, _ = parser.parse_known_args()
    return args.gpu

GPU_ID = parse_args_early()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

import torch
from pathlib import Path

# LLaVA 모듈 import
sys.path.append(str(Path(__file__).parent))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from PIL import Image
import numpy as np


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_token_flow(model_path, model_base=None, image_path=None):
    """토큰 전달 과정 테스트"""
    
    print_section("1. 모델 로드")
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map='auto'
    )
    print(f"✓ 모델 로드 완료: {model_name}")
    if model_base:
        print(f"  - Base model: {model_base}")
    print(f"  - Device: {model.device} (GPU {GPU_ID})")
    print(f"  - dtype: {model.dtype}")
    
    # 모델에 tokenizer 설정 (prepare_inputs_for_summary_generation_batch에서 필요)
    model.tokenizer = tokenizer
    
    # 설정 확인
    print_section("2. 토큰 압축 설정 확인")
    use_summary_tokens = getattr(model.config, 'use_summary_tokens', False)
    num_summary_tokens = getattr(model.config, 'num_summary_tokens', 0)
    kmeans_init = getattr(model.config, 'kmeans_init', False)
    use_dual_lora = getattr(model.config, 'use_dual_lora', False)
    
    print(f"✓ use_summary_tokens: {use_summary_tokens}")
    print(f"✓ num_summary_tokens: {num_summary_tokens}")
    print(f"✓ kmeans_init: {kmeans_init}")
    print(f"✓ use_dual_lora: {use_dual_lora}")
    
    if not use_summary_tokens:
        print("\n⚠️  use_summary_tokens=False")
        print("   two-stage forward를 테스트하려면 use_summary_tokens=True로 설정하세요.")
        return
    
    # 테스트 이미지 준비
    print_section("3. 테스트 이미지 준비")
    
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert('RGB')
        print(f"✓ 이미지 로드: {image_path}")
    else:
        # 더미 이미지 생성
        image = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
        print("✓ 더미 이미지 생성 (336x336)")
    
    print(f"  - 크기: {image.size}")
    
    # 이미지 전처리
    from llava.mm_utils import process_images
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    
    # 배치 차원 추가
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    print(f"✓ 전처리 완료: {image_tensor.shape}")
    
    # 1st Forward 테스트
    print_section("4. 1st Forward - 대표 토큰 생성")
    
    with torch.no_grad():
        # 실제 사용되는 고정 프롬프트 확인
        from llava.constants import SUMMARY_PROMPT
        
        # 실제 프롬프트 토큰 개수 확인
        real_prompt_tokens = tokenizer.encode(SUMMARY_PROMPT, add_special_tokens=False)
        num_real_prompt_tokens = len(real_prompt_tokens)
        
        print(f"✓ 1st forward에서 사용되는 고정 프롬프트:")
        print(f"  - 텍스트: '{SUMMARY_PROMPT}'")
        print(f"  - 토큰 개수: {num_real_prompt_tokens}")
        print(f"  - 토큰 ID: {real_prompt_tokens}")
        print(f"")
        
        # 입력 준비
        result = model.prepare_inputs_for_summary_generation_batch(
            images=image_tensor,
            image_sizes=None,
            return_attention_mask=True
        )
        
        if len(result) == 3:
            inputs_embeds, summary_positions, attention_mask = result
        else:
            inputs_embeds, summary_positions = result
            attention_mask = None
        
        summary_start, summary_end = summary_positions
        num_summary = summary_end - summary_start
        total_seq_len = inputs_embeds.shape[1]
        
        # 토큰 구성 분석
        # [고정 프롬프트] + [이미지 토큰들] + [대표 토큰들]
        num_image_tokens = summary_start - num_real_prompt_tokens
        
        print(f"✓ 1st forward 입력 구성 (정확한 분석):")
        print(f"  📊 총 시퀀스 길이: {total_seq_len}")
        print(f"  📝 고정 프롬프트: {num_real_prompt_tokens}개 토큰 ('{SUMMARY_PROMPT}')")
        print(f"  🖼️  이미지 토큰: {num_image_tokens}개 (vision encoder + projector 출력)")
        print(f"  🎯 대표 토큰: {num_summary}개 (위치 {summary_start}~{summary_end})")
        print(f"  ✅ 검증: {num_real_prompt_tokens} + {num_image_tokens} + {num_summary} = {total_seq_len}")
        print(f"")
        
        # 검증
        expected_len = num_real_prompt_tokens + num_image_tokens + num_summary
        if total_seq_len == expected_len:
            print(f"  ✅ 시퀀스 구성 정확함!")
        else:
            print(f"  ⚠️  시퀀스 길이 불일치! 기대: {expected_len}, 실제: {total_seq_len}")
        print(f"")
        print(f"✓ inputs_embeds: {inputs_embeds.shape}")
        print(f"✓ attention_mask: {'적용' if attention_mask is not None else '없음'}")
        
        # Forward
        forward_kwargs = {
            'inputs_embeds': inputs_embeds,
            'output_hidden_states': True,
            'return_dict': True
        }
        
        if attention_mask is not None:
            from llava.model.attention_utils import combine_masks, convert_mask_to_additive
            batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
            padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)
            combined_mask = combine_masks(attention_mask, padding_mask)
            additive_mask = convert_mask_to_additive(combined_mask, dtype=inputs_embeds.dtype)
            forward_kwargs['attention_mask'] = additive_mask
        
        outputs = model.model(**forward_kwargs)
        
        # Hidden states 추출
        last_hidden_states = outputs.hidden_states[-1]
        summary_hidden_states = model.extract_summary_hidden_states(
            last_hidden_states,
            summary_positions
        )
        
        print(f"✓ 출력:")
        print(f"  - last_hidden_states: {last_hidden_states.shape}")
        print(f"  - summary_hidden_states: {summary_hidden_states.shape}")
        print(f"  - mean: {summary_hidden_states.mean().item():.6f}")
        print(f"  - std: {summary_hidden_states.std().item():.6f}")
        
        # 각 토큰 norm
        print(f"✓ 대표 토큰 norm:")
        for i in range(num_summary):
            norm = summary_hidden_states[0, i].norm().item()
            print(f"  토큰[{i}]: {norm:.4f}")
        
        # 대표 토큰 간 유사도 분석
        if num_summary > 1:
            norm_hidden = summary_hidden_states[0] / summary_hidden_states[0].norm(dim=1, keepdim=True)
            similarity_matrix = torch.mm(norm_hidden, norm_hidden.t())
            
            # 비대각선 평균 유사도
            mask = ~torch.eye(num_summary, dtype=torch.bool, device=similarity_matrix.device)
            off_diagonal_sim = similarity_matrix[mask].mean().item()
            
            print(f"\n✓ 대표 토큰 간 독립성 검증:")
            print(f"  - 평균 cosine similarity (비대각): {off_diagonal_sim:.4f}")
            print(f"  - 범위: [{similarity_matrix[mask].min():.4f}, {similarity_matrix[mask].max():.4f}]")
            
            if off_diagonal_sim > 0.9:
                print(f"  ⚠️  유사도가 너무 높음! 대표 토큰이 독립적이지 않을 수 있음")
            elif off_diagonal_sim < 0.3:
                print(f"  ✅ 대표 토큰이 잘 독립되어 있음 (attention mask 작동)")
            else:
                print(f"  ℹ️  적절한 유사도 (어느정도 독립성 유지)")
    
    # 2nd Forward 테스트
    print_section("5. 2nd Forward - 대표 토큰으로 생성")
    
    with torch.no_grad():
        # 테스트 텍스트 프롬프트
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        
        text_prompt = "What is in this image?"
        prompt_with_image = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
        
        input_ids = tokenizer_image_token(
            prompt_with_image,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        
        print(f"✓ 텍스트 프롬프트 구조:")
        print(f"  - 원본: '{text_prompt}'")
        print(f"  - IMAGE 포함: '{prompt_with_image}'")
        print(f"  - input_ids shape: {input_ids.shape}")
        print(f"  - input_ids: {input_ids[0].tolist()}")
        
        # IMAGE_TOKEN 위치 분석
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        if image_token_mask.any():
            img_pos = image_token_mask.nonzero(as_tuple=True)[1][0].item()
            print(f"  - IMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX})가 위치 {img_pos}에 있음")
            
            # 토큰별 디코딩
            decoded_tokens = []
            for tid in input_ids[0].tolist():
                if tid == IMAGE_TOKEN_INDEX:
                    decoded_tokens.append('<IMAGE>')
                else:
                    decoded_tokens.append(tokenizer.decode([tid]))
            print(f"  - 토큰 분해: {decoded_tokens}")
            print(f"  - 의미: <IMAGE> 위치에 대표 토큰 {num_summary}개가 삽입됨")
        print(f"")
        
        # 2nd forward 입력 준비
        attention_mask_input = torch.ones_like(input_ids, dtype=torch.bool)
        
        _, position_ids, attention_mask_2nd, _, inputs_embeds_2nd, labels_2nd = \
            model.prepare_inputs_with_summary(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=attention_mask_input,
                past_key_values=None,
                labels=None,
                summary_hidden_states=summary_hidden_states,
                image_sizes=None
            )
        
        print(f"✓ 2nd forward 입력:")
        print(f"  - inputs_embeds_2nd: {inputs_embeds_2nd.shape}")
        print(f"  - mean: {inputs_embeds_2nd.mean().item():.6f}")
        print(f"  - std: {inputs_embeds_2nd.std().item():.6f}")
        
        # 대표 토큰이 삽입된 위치 찾기 (IMAGE_TOKEN_INDEX 위치)
        image_token_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)
        if len(image_token_positions[1]) > 0:
            image_token_idx = image_token_positions[1][0].item()
            summary_insert_start = image_token_idx
            summary_insert_end = image_token_idx + summary_hidden_states.shape[1]
            
            print(f"\n✓ 대표 토큰 삽입 위치 검증:")
            print(f"  - IMAGE_TOKEN 위치: {image_token_idx}")
            print(f"  - 대표 토큰 삽입 범위: {summary_insert_start}~{summary_insert_end}")
            print(f"  - 기대 시퀀스 길이: {input_ids.shape[1] - 1 + summary_hidden_states.shape[1]} (텍스트-1 + 대표토큰)")
            print(f"  - 실제 시퀀스 길이: {inputs_embeds_2nd.shape[1]}")
            
            if inputs_embeds_2nd.shape[1] == input_ids.shape[1] - 1 + summary_hidden_states.shape[1]:
                print(f"  ✅ 대표 토큰이 올바르게 삽입됨")
            else:
                print(f"  ⚠️  시퀀스 길이가 예상과 다름!")
        
        # Forward
        outputs = model(
            inputs_embeds=inputs_embeds_2nd,
            attention_mask=attention_mask_2nd,
        )
        
        logits = outputs.logits
        
        print(f"\n✓ 2nd forward 출력:")
        print(f"  - logits: {logits.shape}")
        
        # 다음 토큰 예측
        next_token_logits = logits[0, -1]
        next_token_id = next_token_logits.argmax().item()
        next_token = tokenizer.decode([next_token_id])
        
        print(f"✓ 예측된 다음 토큰: '{next_token}' (ID: {next_token_id})")
    
    # 요약
    print_section("6. 테스트 완료")
    
    # 최종 검증
    print("🔍 최종 검증:")
    print(f"  1. K-means 사용: {'✅' if kmeans_init else '❌'}")
    print(f"  2. 대표 토큰 개수: {num_summary_tokens}개")
    print(f"  3. Attention mask 적용: {'✅' if attention_mask is not None else '❌'}")
    print(f"  4. Dual LoRA 사용: {'✅' if use_dual_lora else '❌'}")
    
    print("\n✅ 모든 단계 정상 작동")
    print(f"✅ 1st forward: 이미지 → {num_summary_tokens}개 대표 토큰 생성")
    print(f"✅ 2nd forward: 대표 토큰 + 텍스트 → 생성")
    
    # 권장사항
    print("\n💡 체크포인트:")
    if off_diagonal_sim > 0.9:
        print("  ⚠️  대표 토큰 간 유사도가 높음 → attention mask 확인 필요")
    if summary_hidden_states.std().item() > 1.5:
        print("  ⚠️  대표 토큰 std가 높음 → 학습 안정성 확인 필요")
    print("")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="토큰 전달 과정 간단 테스트")
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-v1.5-7b-token-compress-kmeans",
                        help="모델 경로 (LoRA 체크포인트)")
    parser.add_argument("--model-base", type=str, default="checkpoints/llava-v1.5-7b",
                        help="Base 모델 경로 (LoRA 사용 시 필요)")
    parser.add_argument("--image", type=str, default=None,
                        help="테스트 이미지 경로 (없으면 더미 이미지 사용)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="사용할 GPU 번호 (기본값: 1)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  토큰 전달 디버깅 테스트")
    print("="*60)
    print(f"\n모델: {args.model_path}")
    if args.model_base:
        print(f"Base: {args.model_base}")
    print(f"이미지: {args.image if args.image else '더미 이미지'}")
    print(f"GPU: {GPU_ID}\n")
    
    try:
        test_token_flow(args.model_path, args.model_base, args.image)
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
