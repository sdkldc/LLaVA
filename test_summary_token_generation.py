"""
이미지 요약 토큰 생성 테스트 스크립트

이 스크립트는 다음을 테스트합니다:
1. 요약 토큰 모듈 생성 및 초기화
2. "Summarize the image in a few words." 프롬프트로 입력 준비
3. LLM Forward를 통한 요약 토큰의 hidden states 추출
"""

import torch
from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.constants import SUMMARY_PROMPT
from PIL import Image
import requests
from io import BytesIO


def test_summary_token_generation():
    """요약 토큰 생성 파이프라인 테스트"""
    
    print("=" * 80)
    print("이미지 요약 토큰 생성 테스트 시작")
    print("=" * 80)
    
    # 1. 모델 및 토크나이저 로드
    print("\n[1단계] 모델 로딩...")
    model_path = "./checkpoints/llava-v1.5-7b"
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava-v1.5-7b"
        )
        print(f"✓ 모델 로드 완료: {model_path}")
        print(f"  - Context length: {context_len}")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return
    
    # 2. 요약 토큰 설정
    print("\n[2단계] 요약 토큰 모듈 설정...")
    num_summary_tokens = 8
    model.config.use_summary_tokens = True
    model.config.num_summary_tokens = num_summary_tokens
    
    # 요약 토큰 모듈 생성
    from llava.model.summary_tokens import build_summary_tokens
    model.model.summary_tokens = build_summary_tokens(model.config)
    model.tokenizer = tokenizer  # 토크나이저 설정
    
    print(f"✓ 요약 토큰 모듈 생성 완료")
    print(f"  - 요약 토큰 개수: {num_summary_tokens}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    
    # 3. 테스트 이미지 로드
    print("\n[3단계] 테스트 이미지 로드...")
    try:
        # 샘플 이미지 URL (COCO 이미지)
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✓ 이미지 로드 완료: {image.size}")
    except Exception as e:
        print(f"✗ 이미지 로드 실패: {e}")
        print("  대신 랜덤 이미지 텐서 사용...")
        # 더미 이미지 생성
        image = Image.new('RGB', (336, 336), color='red')
    
    # 이미지 전처리
    from llava.mm_utils import process_images
    images_tensor = process_images([image], image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    
    print(f"  - 이미지 텐서 shape: {images_tensor.shape}")
    
    # 4. 요약 토큰 생성 입력 준비
    print("\n[4단계] 요약 토큰 생성을 위한 입력 준비...")
    print(f"  - 고정 프롬프트: '{SUMMARY_PROMPT}'")
    
    try:
        with torch.no_grad():
            # 입력 준비
            inputs_embeds, summary_positions = model.prepare_inputs_for_summary_generation_batch(
                images=images_tensor,
                image_sizes=None,
                return_attention_mask=False
            )
            
            print(f"✓ 입력 임베딩 준비 완료")
            print(f"  - inputs_embeds shape: {inputs_embeds.shape}")
            print(f"  - 요약 토큰 위치: {summary_positions}")
            
            # 프롬프트, 이미지, 요약 토큰 길이 확인
            prompt_tokens = tokenizer.encode(SUMMARY_PROMPT, add_special_tokens=False)
            print(f"  - 프롬프트 토큰 수: {len(prompt_tokens)}")
            print(f"  - 이미지 토큰 수: {summary_positions[0] - len(prompt_tokens)}")
            print(f"  - 요약 토큰 수: {summary_positions[1] - summary_positions[0]}")
    
    except Exception as e:
        print(f"✗ 입력 준비 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. LLM Forward 실행 및 hidden states 추출
    print("\n[5단계] LLM Forward 실행...")
    
    try:
        with torch.no_grad():
            # Forward pass
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 마지막 레이어의 hidden states
            last_hidden_states = outputs.hidden_states[-1]
            print(f"✓ LLM Forward 완료")
            print(f"  - 마지막 hidden states shape: {last_hidden_states.shape}")
            
            # 요약 토큰의 hidden states 추출
            summary_hidden_states = model.extract_summary_hidden_states(
                last_hidden_states, 
                summary_positions
            )
            
            print(f"✓ 요약 토큰 hidden states 추출 완료")
            print(f"  - Summary hidden states shape: {summary_hidden_states.shape}")
            print(f"  - 예상 shape: [1, {num_summary_tokens}, {model.config.hidden_size}]")
            
            # 검증
            assert summary_hidden_states.shape[0] == 1, "배치 크기가 1이어야 함"
            assert summary_hidden_states.shape[1] == num_summary_tokens, f"요약 토큰 개수가 {num_summary_tokens}개여야 함"
            assert summary_hidden_states.shape[2] == model.config.hidden_size, "Hidden size 불일치"
            
            print("\n✓✓✓ 모든 검증 통과! ✓✓✓")
            
            # 요약 hidden states 통계
            print("\n[요약 Hidden States 통계]")
            print(f"  - Mean: {summary_hidden_states.mean().item():.6f}")
            print(f"  - Std: {summary_hidden_states.std().item():.6f}")
            print(f"  - Min: {summary_hidden_states.min().item():.6f}")
            print(f"  - Max: {summary_hidden_states.max().item():.6f}")
            
    except Exception as e:
        print(f"✗ Forward 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_summary_token_generation()
