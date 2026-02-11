"""
학습 로직을 변경하지 않고 경고만 억제하는 안전한 패치
- 동작 방식은 그대로 유지 (기존 체크포인트와 호환)
- 경고 메시지만 숨김
"""

import warnings


def suppress_training_warnings():
    """
    학습 중 발생하는 불필요한 경고만 억제 (동작 변경 없음)
    """
    # 1. torch.utils.checkpoint use_reentrant 경고
    warnings.filterwarnings(
        'ignore', 
        message='.*use_reentrant.*',
        category=UserWarning,
        module='torch.utils.checkpoint'
    )
    
    # 2. requires_grad=True 경고 (정상 동작이므로 무시)
    warnings.filterwarnings(
        'ignore',
        message='.*None of the inputs have requires_grad=True.*',
        category=UserWarning,
        module='torch.utils.checkpoint'
    )
    
    # 3. state_dict positional args 경고
    warnings.filterwarnings(
        'ignore',
        message='.*Positional args are being deprecated.*',
        category=UserWarning,
        module='torch.nn.modules.module'
    )
    
    # 4. DeepSpeed stage3 cache flush 경고 (메모리 압박 시 정상)
    warnings.filterwarnings(
        'ignore',
        message='.*pytorch allocator cache flushes.*',
        category=UserWarning
    )
    
    print("✓ Training warnings suppressed (동작 변경 없음, 경고만 숨김)")


def apply_all_patches():
    """
    안전한 경고 억제 패치만 적용
    """
    suppress_training_warnings()
