#!/usr/bin/bash
source config.sh

# ─────────────────── SLURM 헤더 ───────────────────
#SBATCH -J $SLURM_JOB_NAME
#SBATCH --gres=gpu:$SLURM_GPU
#SBATCH --cpus-per-gpu=$SLURM_CPUS
#SBATCH --mem-per-gpu=$SLURM_MEM
#SBATCH -p $SLURM_PART
#SBATCH -w $SLURM_NODE
#SBATCH -t $SLURM_TIME
#SBATCH -o $LOGS_DIR/slurm-%A.out

#### ▼ 여기에 생성할 유튜브 URL을 넣어주세요 ▼
YOUTUBE_URL="https://youtu.be/qHHbhVDaFqY?si=KBquXqofl5Xewj2u"

# 🧠 Conda 환경 설정
source $CONDA_DIR
# Conda 환경 활성화
conda activate $CONDA_ENV

# 📁 작업 디렉토리 이동
cd $REPO_DIR

# 🔧 데이터셋 생성 실행
python train/generate_dataset.py "$YOUTUBE_URL"

# ✅ 완료 메시지
echo "[✅] 데이터셋 생성 완료: $YOUTUBE_URL"

exit 0
