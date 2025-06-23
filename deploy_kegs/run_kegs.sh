#!/usr/bin/bash

#SBATCH -J KEGS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=10G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g6
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out

echo "🎮 KEGS 서버 자동 실행 시작"

# (1) Java
export JAVA_HOME=/data/lch0324/downloads/jdk-17
export PATH=$JAVA_HOME/bin:$PATH

# (2) Conda 환경
source /data/lch0324/anaconda3/etc/profile.d/conda.sh
conda activate kegs_env

export PATH=$CONDA_PREFIX/bin:$PATH

export FONTCONFIG_PATH=/home/lch0324/.fonts
export TORCH_HOME=/data/lch0324/.cache/torch
export KEGS_ENV="SLURM 환경"

export PATH=/data/lch0324/downloads/ffmpeg-static/ffmpeg-7.0.2-amd64-static:$PATH

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
