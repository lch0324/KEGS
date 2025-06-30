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

# 🧠 Conda 환경 설정
source $CONDA_DIR
# Conda 환경 활성화
conda activate $CONDA_ENV

# 📁 작업 디렉토리 이동
cd $REPO_DIR

# 🔧 파인튜닝 실행
python train/finetune.py

# ✅ 완료 메시지
echo "[✅] Whisper 모델 파인튜닝 완료"

exit 0