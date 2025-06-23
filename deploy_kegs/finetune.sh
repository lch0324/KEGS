#!/usr/bin/bash

#SBATCH -J KEGS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out

# 🧠 Conda 환경 설정
source /data/lch0324/anaconda3/etc/profile.d/conda.sh
conda activate kegs_env

# 📁 작업 디렉토리 이동 (필요 시 수정)
cd /data/lch0324/repos/kegs

# 🔧 파인튜닝 실행
python train/finetune.py

# ✅ 완료 메시지
echo "[✅] Whisper 모델 파인튜닝 완료"

exit 0