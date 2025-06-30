# 🎮 KEGS (Korean Enhanced Game Subtitles)

KEGS는 Whisper 기반 로컬 모델을 finetuning하여 게임 영상에서 한국어 자막을 생성하고 영상을 송출하는 웹 서비스입니다.

## 📺 프로젝트 소개

- 🔧**train**: Whisper 기반 모델을 사용하여 데이터셋 생성 및 finetuning
- 🧠**backend**: 로컬 및 ssh 접속 환경에서 자막을 생성하고 영상에 렌더링
- 🧩**frontend**: 유튜브 링크를 입력하여 자막이 렌더링된 영상을 웹에서 스트리밍

## 🔍 주요 특징

- **로컬 처리**: `/generate` 엔드포인트로 로컬에서 자막 생성 및 자막 입힌 영상 송출
- **원격 처리**: `/process` 엔드포인트로 SSH SLURM 클러스터에서 자막 생성 후 결과 영상 송출
- **웹 UI**: 단일 HTML/CSS/JavaScript 인터페이스에서 URL 입력 및 버튼 클릭으로 처리
- **로컬 Whisper 기반 모델**을 finetuning하여 정확도 높은 한국어 자막 생성
- 데이터셋 자동 생성 및 QLoRA 기반 효율적 미세조정
- **음성 구간 감지**: Silero VAD로 음성 구간만 추출해 정확도 향상
- **자연어 분할**: Komoran 형태소 분석 기반 의미 단위 자막 분할

## 디렉터리 구조

```
KEGS/
├─ backend/                     # FastAPI 백엔드 코드
│  ├─ main.py                   # 서버 엔트리 포인트 (로컬/원격)
│  ├─ subtitle_generator.py     # SRT 자막 생성 로직
│  ├─ video_renderer.py         # 영상 다운로드 및 ffmpeg 자막 입힘
│  └─ config.py                 # 환경 변수 설정
├─ deploy_kegs/                 # 원격 서버용 배치 스크립트
│  ├─ backend/                  # 원격 백엔드 코드
│  │  ├─ subtitle_generator.py  # 원격 SRT 자막 생성 로직
│  │  └─ config.py              # 환경 변수 설정
│  ├─ main.py                   # 링크 감시 및 SRT 처리
│  ├─ run_kegs.sh               # SLURM 실행 스크립트
│  ├─ temp/                     # 원격 임시 파일
│  ├─ logs/                     # 원격 실행 로그 및 에러
│  ├─ inputs/                   # 원격 링크 업로드
│  └─ outputs/                  # 원격 처리 결과
├─ frontend/                    # 웹 UI 파일
│  ├─ index.html                # HTML 템플릿
│  ├─ style.css                 # 스타일 시트
│  └─ main.js                   # 클라이언트 자바스크립트
├─ train/                       # 데이터셋 생성 및 모델 파인튜닝
│  ├─ generate_dataset.py       # YouTube → WAV 및 TSV 데이터 생성
│  ├─ finetune.py               # Whisper 모델 파인튜닝
│  └─ config.json               # 파인튜닝 설정
├─ temp/                        # 임시 파일
├─ logs/                        # 실행 로그 및 에러
├─ launch.bat                   # 로컬 서버 및 UI 동시 실행 스크립트
├─ requirements.txt             # Python 의존성
└─ README.md                    # 프로젝트 설명
```


## 설치 및 실행

1. **저장소 클론 및 이동**

   ```bash
   git clone https://github.com/lch0324/KEGS.git
   cd KEGS
   ```

2. **환경 설정**

   * `backend/config.example.py`를 복사하여 `config.py`로 사용
      * 모델 경로와 ssh 접속 시 ssh 정보 입력
   * `train/config.json`을 열어 `base_model`, `output_dir`, `dataset_path` 확인

3. **가상환경 생성 및 활성화**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: .\\venv\\Scripts\\activate
   ```

4. **의존성 설치**

   ```bash
   pip install -r requirements.txt
   ```

5. **로컬 실행**

   * `launch.bat` 실행 (Windows):

     ```bat
     launch.bat
     ```
   * 수동 실행:

     1. 백엔드 서버

        ```bash
        uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
        ```
     2. 프론트엔드

        * VSCode Live Server 등으로 `frontend/index.html` 열기 (기본 포트 5500)

6. **웹 브라우저 접속**

   * `http://localhost:5500` 으로 UI 열기

7. **사용 방법**

   1. YouTube 영상 링크 입력
   2. **자막 영상 생성** 클릭 → 로컬 처리(`/generate`) → 영상 스트리밍
   3. **원격 자막 영상 생성** 클릭 → 원격 처리(`/process`) → SSH SLURM 처리 후 결과 스트리밍


## 설정 파일 예시

* `backend/config.example.py`

  ```python
  SSH_HOST = "aurora.khu.ac.kr"
  SSH_PORT = 30080
  SSH_USER = "lch0324"
  SSH_PASSWORD = "<YOUR_PASSWORD>"

  REMOTE_INPUT_DIR = "/data/lch0324/repos/kegs/inputs"
  REMOTE_OUTPUT_DIR = "/data/lch0324/repos/kegs/outputs"
  ```

* `train/config.json`

  ```json
  {
    "base_model": "openai/whisper-large-v3-turbo",
    "output_dir": "./models/whisper-large-v3-turbo-finetuned",
    "dataset_path": "./dataset"
  }
  ```

## 배치 처리 (원격)

* `deploy_kegs/run_kegs.sh`를 사용하여 SLURM 클러스터에 작업 제출

  ```bash
  cd deploy_kegs
  sbatch run_kegs.sh
  ```

## 라이선스

* MIT 라이선스