#!/usr/bin/bash

#SBATCH -J KEGS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out

#### ▼ 여기에 생성할 유튜브 URL을 넣어주세요 ▼
YOUTUBE_URL="https://youtu.be/qHHbhVDaFqY?si=KBquXqofl5Xewj2u"

# 🧠 Conda 환경 설정
source /data/lch0324/anaconda3/etc/profile.d/conda.sh
conda activate kegs_env

# 📁 작업 디렉토리 이동
cd /data/lch0324/repos/kegs

# 🔧 데이터셋 생성 실행
python train/generate_dataset.py "$YOUTUBE_URL"

# ✅ 완료 메시지
echo "[✅] 데이터셋 생성 완료: $YOUTUBE_URL"

exit 0
