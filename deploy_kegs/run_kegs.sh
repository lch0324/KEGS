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

echo "🎮 KEGS 서버 자동 실행 시작"

# (1) Java
export JAVA_HOME=$JAVA_HOME
export PATH=$JAVA_HOME/bin:$PATH

# (2) Conda 환경
source $CONDA_DIR
# Conda 환경 활성화
conda activate $CONDA_ENV

export PATH=$CONDA_PREFIX/bin:$PATH

export FONTCONFIG_PATH=$FONTCONFIG_PATH
export TORCH_HOME=$TORCH_HOME
export KEGS_ENV="SLURM 환경"

export PATH=$FFMPEG_BIN:$PATH

# ✅ 경로 확인
which ffmpeg
ffmpeg -version

link_file=$(ls inputs/*.txt | head -n 1)
if [ -z "$link_file" ]; then
    echo "[❌] 링크 파일 없음. 종료."
    exit 1
fi

python3 main.py
exit 0
