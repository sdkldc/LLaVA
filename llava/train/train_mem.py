from llava.train.train import train # trian.py의 train 함수 임포트

if __name__ == "__main__": # 해당 파일이 직접 실행될 때, train 함수 호출
    # eager attention 사용: 커스텀 attention mask 지원 (요약 토큰 간 상호 참조 방지)
    # Flash Attention 2는 커스텀 마스크를 지원하지 않아 방법론 구현 불가
    train(attn_implementation="eager")
