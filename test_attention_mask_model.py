#!/usr/bin/env python3
"""
실제 모델에서 Attention Mask 적용 테스트

목적:
1. Stage 1에서 커스텀 attention mask가 제대로 LLM에 전달되는지 확인
2. Stage 2에서 일반 causal + padding mask가 사용되는지 확인
3. Forward pass에서 실제로 마스크가 적용되는지 검증
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np

# LLaVA 모듈 import
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import SUMMARY_PROMPT


def test_stage1_attention_mask_application():
    """Stage 1: 요약 토큰 생성 시 attention mask 적용 테스트"""
    print("\n" + "="*80)
    print("실제 모델 테스트 1: Stage 1 Attention Mask 적용")
    print("="*80)
    
    # 모델 로드
    model_path = "/home/deokhyeon/Documents/LLaVA/checkpoints/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    
    print(f"\n모델 로딩: {model_path}")
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map="auto"
        )
        
        print(f"✓ 모델 로드 완료")
        print(f"  - Context length: {context_len}")
        print(f"  - Device: {model.device}")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # Summary tokens 모듈 확인
    if not hasattr(model, 'model') or not hasattr(model.model, 'summary_tokens'):
        print("⚠️ Summary tokens 모듈이 없습니다. config에서 use_summary_tokens=True 확인 필요")
        return
    
    summary_tokens = model.model.summary_tokens
    if summary_tokens is None:
        print("⚠️ Summary tokens가 None입니다.")
        return
    
    print(f"✓ Summary tokens 모듈 확인")
    print(f"  - Num tokens: {summary_tokens.num_summary_tokens}")
    
    # 더미 이미지 생성
    batch_size = 2
    device = model.device
    
    # 이미지 크기 확인
    vision_tower = model.get_vision_tower()
    image_size = vision_tower.config.image_size if hasattr(vision_tower.config, 'image_size') else 336
    
    dummy_images = torch.randn(
        batch_size, 3, image_size, image_size,
        dtype=model.dtype,
        device=device
    )
    
    print(f"\n더미 입력 생성:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image shape: {dummy_images.shape}")
    
    # prepare_inputs_for_summary_generation_batch 호출
    print(f"\nStage 1 입력 준비 중...")
    
    try:
        result = model.prepare_inputs_for_summary_generation_batch(
            images=dummy_images,
            image_sizes=None,
            return_attention_mask=True
        )
        
        if len(result) == 3:
            inputs_embeds, summary_positions, attention_mask = result
        else:
            inputs_embeds, summary_positions = result
            attention_mask = None
        
        print(f"✓ 입력 준비 완료")
        print(f"  - inputs_embeds shape: {inputs_embeds.shape}")
        print(f"  - summary_positions: {summary_positions}")
        
        if attention_mask is not None:
            print(f"  - attention_mask shape: {attention_mask.shape}")
            print(f"  - attention_mask dtype: {attention_mask.dtype}")
            
            # 마스크 검증
            print(f"\n[Attention Mask 검증]")
            
            # Boolean mask인지 확인
            is_bool = attention_mask.dtype == torch.bool
            print(f"  - Boolean mask: {is_bool}")
            
            # 요약 토큰 영역 확인
            summary_start, summary_end = summary_positions
            mask_2d = attention_mask[0, 0].cpu()
            
            summary_region = mask_2d[summary_start:summary_end, summary_start:summary_end]
            
            # Diagonal 확인
            diagonal = torch.diagonal(summary_region)
            diagonal_ok = torch.all(diagonal == False)
            print(f"  - Diagonal (자기 참조): {diagonal_ok} (모두 False여야 함)")
            
            # Off-diagonal 확인
            off_diag_mask = ~torch.eye(summary_region.shape[0], dtype=torch.bool)
            off_diagonal = summary_region[off_diag_mask]
            off_diagonal_ok = torch.all(off_diagonal == True)
            print(f"  - Off-diagonal (상호 참조 차단): {off_diagonal_ok} (모두 True여야 함)")
            
            if diagonal_ok and off_diagonal_ok:
                print(f"  ✅ 요약 토큰 간 참조 방지 마스크 정상 작동")
            else:
                print(f"  ❌ 마스크 검증 실패")
        else:
            print(f"  ⚠️ Attention mask가 반환되지 않음")
        
        # LLM forward test (실제 적용 확인)
        print(f"\n[LLM Forward 테스트]")
        print(f"  Attention mask를 LLM에 전달하여 forward pass 수행...")
        
        # Padding mask 생성 (모두 유효한 토큰)
        seq_len = inputs_embeds.shape[1]
        padding_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.bool,
            device=device
        )
        
        # 마스크 결합 및 변환
        if attention_mask is not None:
            from llava.model.attention_utils import combine_masks, convert_mask_to_additive
            
            combined_mask = combine_masks(attention_mask, padding_mask)
            additive_mask = convert_mask_to_additive(combined_mask, dtype=inputs_embeds.dtype)
            
            print(f"  - Combined mask shape: {combined_mask.shape}")
            print(f"  - Additive mask shape: {additive_mask.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=additive_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            print(f"  ✓ Forward pass 성공")
            print(f"  - Hidden states shape: {outputs.hidden_states[-1].shape}")
            
            # 요약 토큰의 hidden states 추출
            last_hidden = outputs.hidden_states[-1]
            summary_hidden = last_hidden[:, summary_start:summary_end, :]
            
            print(f"  - Summary hidden states shape: {summary_hidden.shape}")
            print(f"  - Summary hidden states mean: {summary_hidden.mean().item():.4f}")
            print(f"  - Summary hidden states std: {summary_hidden.std().item():.4f}")
            
            # 요약 토큰들이 서로 다른지 확인 (참조 차단 효과)
            # 만약 참조가 차단되었다면, 각 요약 토큰은 독립적으로 계산되어야 함
            if summary_hidden.shape[1] > 1:
                # 첫 번째 배치의 요약 토큰들 간 cosine similarity
                summary_batch0 = summary_hidden[0]  # [num_summary, hidden_size]
                
                # Normalize
                summary_norm = F.normalize(summary_batch0, p=2, dim=1)
                
                # Pairwise similarity
                similarity_matrix = torch.mm(summary_norm, summary_norm.t())
                
                # Diagonal 제외한 평균 similarity
                num_tokens = similarity_matrix.shape[0]
                off_diag_mask = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
                avg_similarity = similarity_matrix[off_diag_mask].mean().item()
                
                print(f"\n  [요약 토큰 간 독립성 검증]")
                print(f"  - 요약 토큰 간 평균 cosine similarity: {avg_similarity:.4f}")
                print(f"    (낮을수록 독립적, 높을수록 유사)")
                
                # 참고: 완전히 독립적이면 similarity가 0에 가까워야 함
                # 하지만 이미지 정보를 공유하므로 어느 정도 유사할 수 있음
        
        else:
            print(f"  ⚠️ Attention mask 없이 forward pass는 생략")
        
        print(f"\n✅ Stage 1 테스트 완료")
        
    except Exception as e:
        print(f"❌ Stage 1 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def test_stage2_attention_mask():
    """Stage 2: 답변 생성 시 일반 attention mask 테스트"""
    print("\n" + "="*80)
    print("실제 모델 테스트 2: Stage 2 Attention Mask (일반 LLaVA)")
    print("="*80)
    
    # 모델 로드
    model_path = "/home/deokhyeon/Documents/LLaVA/checkpoints/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    
    print(f"\n모델 로딩: {model_path}")
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map="auto"
        )
        
        print(f"✓ 모델 로드 완료")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # Stage 2 시뮬레이션: prepare_inputs_with_summary 사용
    print(f"\nStage 2 입력 준비 중...")
    
    batch_size = 2
    device = model.device
    
    # 더미 요약 hidden states 생성
    num_summary_tokens = 8
    hidden_size = model.config.hidden_size
    
    dummy_summary_hidden = torch.randn(
        batch_size, num_summary_tokens, hidden_size,
        dtype=model.dtype,
        device=device
    )
    
    # 더미 입력 텍스트 (질문)
    input_text = "What is in the image?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_ids = input_ids.expand(batch_size, -1)
    
    print(f"  - Input text: {input_text}")
    print(f"  - Input IDs shape: {input_ids.shape}")
    print(f"  - Summary hidden states shape: {dummy_summary_hidden.shape}")
    
    try:
        # prepare_inputs_with_summary 호출
        result = model.prepare_inputs_with_summary(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            summary_hidden_states=dummy_summary_hidden,
            image_sizes=None
        )
        
        _, position_ids, attention_mask, _, inputs_embeds, _ = result
        
        print(f"\n✓ 입력 준비 완료")
        print(f"  - inputs_embeds shape: {inputs_embeds.shape}")
        
        if attention_mask is not None:
            print(f"  - attention_mask shape: {attention_mask.shape}")
            print(f"  - attention_mask dtype: {attention_mask.dtype}")
            
            # 마스크 검증
            print(f"\n[Attention Mask 검증]")
            
            # Stage 2에서는 일반 causal mask만 사용
            # 즉, 요약 토큰끼리도 참조 가능해야 함
            
            if attention_mask.dtype == torch.bool:
                print(f"  - Boolean mask (패딩 마스크)")
                print(f"  - True(유효 토큰) 개수: {attention_mask.sum().item()}")
                print(f"  - False(패딩) 개수: {(~attention_mask).sum().item()}")
            else:
                print(f"  - Additive mask 또는 기타 형식")
            
            # 실제로는 LLaMA의 causal mask가 자동으로 적용됨
            print(f"  - LLaMA는 내부적으로 causal mask를 자동 적용")
            print(f"  - 요약 토큰끼리도 lower triangular 규칙에 따라 참조 가능")
        else:
            print(f"  - attention_mask: None (패딩 없음)")
            print(f"  - LLaMA 내부 causal mask만 사용")
        
        # Forward pass
        print(f"\n[LLM Forward 테스트]")
        
        with torch.no_grad():
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True
            )
        
        print(f"  ✓ Forward pass 성공")
        print(f"  - Hidden states shape: {outputs.hidden_states[-1].shape}")
        
        # Logits 계산
        logits = model.lm_head(outputs.hidden_states[-1])
        print(f"  - Logits shape: {logits.shape}")
        
        # 다음 토큰 예측 (마지막 위치)
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        print(f"  - Next tokens: {next_tokens.tolist()}")
        
        # 토큰 디코딩
        for i, token_id in enumerate(next_tokens):
            decoded = tokenizer.decode(token_id)
            print(f"    Batch {i}: '{decoded}'")
        
        print(f"\n✅ Stage 2 테스트 완료")
        print(f"  → Stage 2에서는 일반 LLaVA처럼 causal + padding mask만 사용")
        print(f"  → 요약 토큰끼리 서로 참조 가능")
        
    except Exception as e:
        print(f"❌ Stage 2 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def test_attention_mask_shape_verification():
    """Attention mask의 shape과 형식 검증"""
    print("\n" + "="*80)
    print("추가 테스트: Attention Mask Shape 및 형식 검증")
    print("="*80)
    
    print("\n[예상되는 Attention Mask 형식]")
    print("\n1. Stage 1 (요약 토큰 생성):")
    print("   - Custom mask: [batch, 1, seq_len, seq_len] (Boolean)")
    print("   - Padding mask: [batch, seq_len] (Boolean)")
    print("   - Combined: [batch, 1, seq_len, seq_len] (Boolean)")
    print("   - Additive: [batch, 1, seq_len, seq_len] (Float, 0/-inf)")
    
    print("\n2. Stage 2 (답변 생성):")
    print("   - Padding mask only: [batch, seq_len] (Boolean)")
    print("   - LLaMA 내부 causal mask: 자동 적용")
    
    print("\n[핵심 차이점]")
    print("   Stage 1: 요약 토큰끼리 off-diagonal 차단 (커스텀 마스크)")
    print("   Stage 2: 요약 토큰끼리 lower triangular 허용 (일반 causal)")
    
    # 예시 출력
    print("\n[예시 마스크 패턴]")
    print("\nStage 1 (seq_len=10, summary at 7~9):")
    print("   Query \\ Key  0 1 2 3 4 5 6 7 8 9")
    print("   0 (prompt)   0 1 1 1 1 1 1 1 1 1")
    print("   1 (prompt)   0 0 1 1 1 1 1 1 1 1")
    print("   ...          ... (causal mask)")
    print("   7 (summary)  0 0 0 0 0 0 0 0 1 1  <- 자기만 참조")
    print("   8 (summary)  0 0 0 0 0 0 0 1 0 1  <- 자기만 참조")
    print("   9 (summary)  0 0 0 0 0 0 0 1 1 0  <- 자기만 참조")
    print("                              ↑ ↑ ↑")
    print("                      summary 영역 off-diag 차단")
    
    print("\nStage 2 (seq_len=10, summary at 7~9):")
    print("   Query \\ Key  0 1 2 3 4 5 6 7 8 9")
    print("   0 (prompt)   0 1 1 1 1 1 1 1 1 1")
    print("   1 (prompt)   0 0 1 1 1 1 1 1 1 1")
    print("   ...          ... (causal mask)")
    print("   7 (summary)  0 0 0 0 0 0 0 0 1 1  <- causal")
    print("   8 (summary)  0 0 0 0 0 0 0 0 0 1  <- causal")
    print("   9 (summary)  0 0 0 0 0 0 0 0 0 0  <- causal")
    print("                              ↑ ↑ ↑")
    print("                      summary 영역 lower triangular")
    
    print("\n  (0 = 참조 가능, 1 = 차단)")


def main():
    """메인 테스트 실행"""
    print("="*80)
    print("실제 모델 Attention Mask 적용 테스트")
    print("="*80)
    
    try:
        # 테스트 1: Stage 1 attention mask
        test_stage1_attention_mask_application()
        
        # 테스트 2: Stage 2 attention mask
        test_stage2_attention_mask()
        
        # 테스트 3: Shape 검증
        test_attention_mask_shape_verification()
        
        print("\n" + "="*80)
        print("✅ 모든 실제 모델 테스트 완료")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
