# 📄 deploy_kegs/config.sh - 스크립트 파일 변수 설정

SLURM_JOB_NAME=KEGS
SLURM_GPU=1
SLURM_CPUS=8
SLURM_MEM=32G
SLURM_PART=batch_ugrad
SLURM_NODE=aurora-g6
SLURM_TIME=1-00:00:00

CONDA_ENV=kegs_env
CONDA_DIR=/data/lch0324/anaconda3/etc/profile.d/conda.sh
REPO_DIR=/data/lch0324/repos/kegs
LOGS_DIR=logs

JAVA_HOME=/data/lch0324/downloads/jdk-17
FONTCONFIG_PATH=/home/lch0324/.fonts
TORCH_HOME=/data/lch0324/.cache/torch
FFMPEG_BIN=/data/lch0324/downloads/ffmpeg-static/ffmpeg-7.0.2-amd64-static