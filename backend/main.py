# 📄 backend/main.py - FastAPI 서버 (로컬: /generate + 원격: /process)

import paramiko
import uuid
import os
import traceback
import time
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from moviepy import VideoFileClip
import config
from video_renderer import render_video_with_subtitles
from subtitle_generator import generate_srt_from_video

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class VideoRequest(BaseModel):
    youtube_url: str

class YouTubeRequest(BaseModel):
    youtube_url: str

# 서버 SFTP 정보
SSH_HOST = config.SSH_HOST
SSH_PORT = config.SSH_PORT
SSH_USER = config.SSH_USER
SSH_PASSWORD = config.SSH_PASSWORD

REMOTE_INPUT_DIR = config.REMOTE_INPUT_DIR
REMOTE_OUTPUT_DIR = config.REMOTE_OUTPUT_DIR

TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

# /generate (로컬)
@app.post("/generate")
def generate_subtitled_video(req: VideoRequest):
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(TEMP_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 유튜브 다운로드
        video_path = render_video_with_subtitles(req.youtube_url, None, output_dir, download_only=True)
        # 2. 자막 생성
        srt_path = os.path.join(output_dir, "subtitles.srt")
        generate_srt_from_video(video_path, srt_path)
        # 3. 자막 입힌 영상 생성
        final_video_path = render_video_with_subtitles(req.youtube_url, srt_path, output_dir, download_only=False)

        # 4. 소요 시간 및 로그 기록
        end_time = time.time()
        consumed_time = int(end_time - start_time)

        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()

        minutes, seconds = divmod(video_duration, 60)
        video_length_str = f"{int(minutes)}:{int(seconds):02d}"
        method = os.getenv("KEGS_ENV", "로컬 환경")

        log_path = os.path.join(LOG_DIR, "consumed_time.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{video_id}]\n")
            f.write(f"시작 시간: {start_datetime}\n")
            f.write(f"방법: {method}\n")
            f.write(f"영상 길이: [{video_length_str}]\n")
            f.write(f"소요 시간: {int(consumed_time // 60)}분 {int(consumed_time % 60)}초\n\n")

        # mp4 파일 반환
        return FileResponse(final_video_path, media_type="video/mp4", filename="output.mp4")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        os.makedirs(LOG_DIR, exist_ok=True)
        backend_error_log = os.path.join(LOG_DIR, "backend_error.txt")
        with open(backend_error_log, "a", encoding="utf-8") as log_file:
            log_file.write(error_msg + "\n")

        return {"error": str(e)}

# /process (원격)
def upload_link_to_server(video_id, youtube_url):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    sftp = ssh.open_sftp()
    remote_file_path = os.path.join(REMOTE_INPUT_DIR, f"{video_id}.txt").replace("\\", "/")
    with sftp.file(remote_file_path, "w", -1) as f:
        f.write(youtube_url)
        f.flush()
    sftp.close()
    ssh.close()

def submit_sbatch_job():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    command = f"cd /data/lch0324/repos/kegs && sbatch run_kegs.sh"
    ssh.exec_command(command)
    ssh.close()

def download_from_server(remote_path, local_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()

@app.post("/process")
def process_youtube(req: YouTubeRequest):
    try:
        youtube_url = req.youtube_url
        if not youtube_url:
            raise HTTPException(status_code=400, detail="YouTube URL이 없습니다.")

        video_id = str(uuid.uuid4())[:8]
        local_video_dir = os.path.join(TEMP_DIR, video_id)
        os.makedirs(local_video_dir, exist_ok=True)

        # 유튜브 링크 서버에 전송
        upload_link_to_server(video_id, youtube_url)

        # sbatch job 제출
        submit_sbatch_job()

        output_srt = f"{video_id}.srt"
        output_log = "consumed_time.txt"

        remote_srt_path = os.path.join(REMOTE_OUTPUT_DIR, output_srt).replace("\\", "/")
        remote_log_path = os.path.join(REMOTE_OUTPUT_DIR, output_log).replace("\\", "/")

        local_srt_path = os.path.join(local_video_dir, output_srt)
        local_log_path = os.path.join(local_video_dir, output_log)

        print("[⌛] 결과 생성 대기 중...")

        timeout = 86400
        interval = 10
        waited = 0

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
        sftp = ssh.open_sftp()

        while waited < timeout:
            try:
                sftp.stat(remote_srt_path)
                sftp.stat(remote_log_path)

                download_from_server(remote_srt_path, local_srt_path)
                download_from_server(remote_log_path, local_log_path)

                sftp.close()
                ssh.close()
                break
            except IOError:
                time.sleep(interval)
                waited += interval

        if waited >= timeout:
            sftp.close()
            ssh.close()
            raise HTTPException(status_code=500, detail="타임아웃: 결과 파일이 생성되지 않았습니다.")

        time.sleep(1)  # 파일 write 완전 완료 대기

        if not os.path.exists(local_srt_path):
            raise HTTPException(status_code=500, detail="로컬에 srt 파일이 존재하지 않습니다.")

        # 로컬에서 유튜브 mp4 다운로드 및 자막 입히기
        final_video_path = render_video_with_subtitles(youtube_url, local_srt_path, local_video_dir, download_only=False)
        original_video_path = os.path.join(local_video_dir, "original_video.mp4")
        # 원본 mp4 삭제
        if os.path.exists(original_video_path):
            os.remove(original_video_path)

        # 최종 자막 입힌 mp4 반환
        return FileResponse(final_video_path, media_type="video/mp4", filename=os.path.basename(final_video_path))

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        os.makedirs(LOG_DIR, exist_ok=True)
        backend_error_log = os.path.join(LOG_DIR, "backend_error.txt")
        with open(backend_error_log, "a", encoding="utf-8") as log_file:
            log_file.write(error_msg + "\n")
        raise HTTPException(status_code=500, detail="서버 처리 중 오류가 발생했습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
