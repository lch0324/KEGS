# KEGS (Korean Enhanced Game Subtitles)

KEGS는 Whisper-large-v3 로컬 모델을 기반으로 게임 영상에서 한국어 자막을 생성하고 영상을 송출하는 웹 서비스입니다.

## 주요 특징

* **로컬 Whisper-large-v3 모델**을 finetuning하여 정확도 높은 한국어 자막 생성
* **KoNLPy의 Komoran 형태소 분석 기반 절 분할** 및 **Silero VAD 음성 구간 감지**로 자연스러운 타임스탬프 배분
* **FastAPI** 기반 백엔드와 프론트엔드(UI) 통합 실행 스크립트
* 데이터셋 자동 생성 및 QLoRA 기반 효율적 미세조정