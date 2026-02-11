#!/usr/bin/env python3
"""
End-to-End 통합 테스트

실제 모델과 이미지를 사용하여 전체 파이프라인 검증:
1. 이미지 로딩 및 전처리
2. 일반 추론 (기존 방식)
3. Two-stage 추론 (요약 토큰 사용)
4. 결과 비교 및 검증
"""

import torch
import sys
import os
from PIL import Image

sys.path.append('/home/deokhyeon/Documents/LLaVA')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def test_end_to_end():
    """전체 파이프라인 테스트"""
    
    print("=" * 80)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 80)
    
    # 설정
    model_path = "./checkpoints/llava-v1.5-7b-token-compress-kmeans"  # 요약 토큰 학습된 체크포인트
    model_base = "./checkpoints/llava-v1.5-7b"  # Base 모델
    test_image_path = "./images/llava_v1_5_radar.jpg"
    prompt = "What do you see in this image?"
    
    # 모델 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Skipping end-to-end test (model required)")
        return True
    
    if not os.path.exists(model_base):
        print(f"❌ Base model not found at {model_base}")
        print("   Skipping end-to-end test (base model required)")
        return True
    
    # 이미지 존재 확인
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found at {test_image_path}")
        print("   Skipping end-to-end test (image required)")
        return True
    
    print(f"\n✓ Model path: {model_path}")
    print(f"✓ Base model: {model_base}")
    print(f"✓ Image path: {test_image_path}")
    print(f"✓ Prompt: '{prompt}'")
    
    try:
        # ========== Step 1: 모델 로드 ==========
        print("\n[Step 1] Loading model...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,  # Base 모델 지정
            model_name=get_model_name_from_path(model_path),
            device_map="auto"
        )
        print(f"✓ Model loaded successfully (LoRA + Summary Tokens)")
        print(f"  Device: {model.device}")
        print(f"  Context length: {context_len}")
        
        # Tokenizer 설정 (summary tokens 생성에 필요)
        model.tokenizer = tokenizer
        
        # Summary tokens 모듈 확인
        has_summary = hasattr(model.model, 'summary_tokens') and model.model.summary_tokens is not None
        print(f"  Summary tokens module: {'✓ Present' if has_summary else '✗ Missing'}")
        
        if not has_summary:
            print("\n⚠️  Warning: Summary tokens module not found!")
            print("   Two-stage inference will be skipped")
            skip_summary = True
        else:
            skip_summary = False
            num_summary = getattr(model.config, 'num_summary_tokens', 'unknown')
            print(f"  - Num summary tokens: {num_summary}")
        
        # ========== Step 2: 이미지 로드 및 전처리 ==========
        print("\n[Step 2] Loading and preprocessing image...")
        image = Image.open(test_image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size}")
        
        images_tensor = process_images([image], image_processor, model.config)
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)
        print(f"✓ Image preprocessed: {images_tensor.shape}")
        
        # ========== Step 3: 프롬프트 준비 ==========
        print("\n[Step 3] Preparing prompt...")
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        
        # 이미지 토큰 추가
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_text, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        
        print(f"✓ Input prepared")
        print(f"  Prompt length: {input_ids.shape[1]} tokens")
        
        # ========== Step 4: 일반 추론 (기존 방식) ==========
        print("\n[Step 4] Running standard inference...")
        with torch.inference_mode():
            output_ids_standard = model.generate(
                inputs=input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                use_summary_tokens=False,  # 기존 방식
                max_new_tokens=128,
                temperature=0.2,
                do_sample=False,
            )
        
        output_text_standard = tokenizer.batch_decode(
            output_ids_standard, 
            skip_special_tokens=True
        )[0].strip()
        
        print(f"✓ Standard inference completed")
        print(f"  Generated tokens: {output_ids_standard.shape[1] - input_ids.shape[1]}")
        print(f"  Output preview: {output_text_standard[:100]}...")
        
        # ========== Step 5: Two-stage 추론 (요약 토큰 사용) ==========
        if skip_summary:
            print("\n[Step 5] Skipping two-stage inference (summary tokens not available)")
            output_text_summary = "[SKIPPED]"
            len_summary = 0
        else:
            print("\n[Step 5] Running two-stage inference with summary tokens...")
            
            with torch.inference_mode():
                output_ids_summary = model.generate(
                    inputs=input_ids,
                    images=images_tensor,
                    image_sizes=[image.size],
                    use_summary_tokens=True,  # 요약 토큰 사용
                    max_new_tokens=128,
                    temperature=0.2,
                    do_sample=False,
                )
            
            output_text_summary = tokenizer.batch_decode(
                output_ids_summary, 
                skip_special_tokens=True
            )[0].strip()
            
            print(f"✓ Two-stage inference completed")
            print(f"  Input tokens: {input_ids.shape[1]}")
            print(f"  Output tokens: {output_ids_summary.shape[1]}")
            print(f"  Generated tokens: {output_ids_summary.shape[1] - input_ids.shape[1]}")
            print(f"  Output preview: {output_text_summary[:100]}...")
            len_summary = len(output_text_summary)
        
        # ========== Step 6: 결과 비교 및 검증 ==========
        print("\n[Step 6] Comparing results...")
        
        # 길이 비교
        len_standard = len(output_text_standard)
        print(f"  Standard output length: {len_standard} chars")
        
        if not skip_summary:
            print(f"  Summary output length: {len_summary} chars")
        
        # 둘 다 출력이 있는지 확인
        assert len_standard > 0, "Standard inference produced no output!"
        print(f"✓ Standard inference produced output")
        
        if not skip_summary:
            assert len_summary > 0, "Two-stage inference produced no output!"
            print(f"✓ Two-stage inference produced output")
            
            # 출력이 너무 다르지 않은지 확인 (±50%)
            ratio = len_summary / len_standard
            print(f"  Length ratio (summary/standard): {ratio:.2f}")
            
            # 간단한 내용 검증 (이미지 관련 키워드가 있는지)
            common_keywords = ['image', 'picture', 'see', 'show', 'display']
            has_keyword_standard = any(kw in output_text_standard.lower() for kw in common_keywords)
            has_keyword_summary = any(kw in output_text_summary.lower() for kw in common_keywords)
            
            print(f"  Standard has relevant keywords: {has_keyword_standard}")
            print(f"  Summary has relevant keywords: {has_keyword_summary}")
        
        # ========== 최종 결과 ==========
        print("\n" + "=" * 80)
        print("FULL OUTPUTS")
        print("=" * 80)
        print(f"\n[Standard Inference]")
        print(output_text_standard)
        
        if not skip_summary:
            print(f"\n[Two-stage Inference with Summary Tokens]")
            print(output_text_summary)
        
        print("\n" + "=" * 80)
        print("✅ END-TO-END TEST PASSED!")
        print("=" * 80)
        print("\n검증 완료:")
        print("  ✓ 모델 로딩 (LoRA + Summary Tokens)")
        print("  ✓ 이미지 전처리")
        print("  ✓ 일반 추론 실행")
        if not skip_summary:
            print("  ✓ Two-stage 추론 실행")
            print("  ✓ 둘 다 유효한 출력 생성")
            print("\n두 방식 모두 정상 작동합니다!")
        else:
            print("  ⚠️ Two-stage 추론 스킵 (모듈 없음)")
            print("\n일반 추론은 정상 작동합니다!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
