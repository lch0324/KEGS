# 📄 deploy_kegs/main.py - 서버용 메인 감시 스크립트

import os
import time
import logging
import subprocess
import datetime
import time
from moviepy import VideoFileClip

from backend.subtitle_generator import generate_srt_from_video
# from backend.video_renderer import render_video_with_subtitles

INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def process_youtube_link(txt_file):
    try:
        logging.info(f"[🚀] 감지된 링크 파일: {txt_file}")

        start_time = time.time()  # 전체 시간 측정 시작
        start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(txt_file, "r", encoding="utf-8") as f:
            youtube_url = f.read().strip()

        if not youtube_url:
            raise ValueError("비어있는 URL 파일")

        # 파일명에서 video_id 추출
        video_id = os.path.splitext(os.path.basename(txt_file))[0]

        local_video_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")
        download_cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4][vcodec!*=av01]",
            "-o", local_video_path.replace("\\", "/"),
            youtube_url
        ]

        logging.info(f"[⬇️] 유튜브 영상 다운로드 시작: {youtube_url}")
        subprocess.run(download_cmd, check=True)
        if not os.path.exists(local_video_path):
            raise FileNotFoundError(f"다운로드된 mp4 파일 없음: {local_video_path}")
        logging.info(f"[✅] 유튜브 다운로드 완료: {local_video_path}")

        srt_path = os.path.join(OUTPUT_DIR, f"{video_id}.srt")
        logging.info("[🧠] SRT 자막 생성 시작")
        generate_srt_from_video(local_video_path, srt_path)
        logging.info(f"[✅] SRT 자막 생성 완료: {srt_path}")

        # output_video_path = os.path.join(OUTPUT_DIR, f"{video_id}_subtitled.mp4")
        # logging.info("[🎞️] 자막 입히기 시작")
        # render_video_with_subtitles(local_video_path, srt_path, output_video_path)
        # logging.info(f"[✅] 자막 입힌 영상 생성 완료: {output_video_path}")

        os.remove(txt_file)
        logging.info(f"[🗑️] 링크 파일 삭제: {txt_file}")

        end_time = time.time()  # ✅ 전체 시간 측정 끝
        consumed_time = int(end_time - start_time)

        # 영상 길이 가져오기
        clip = VideoFileClip(local_video_path)
        video_duration = clip.duration
        clip.close()

        minutes, seconds = divmod(video_duration, 60)
        video_length_str = f"{int(minutes)}:{int(seconds):02d}"

        # 방법 가져오기
        method = os.getenv("KEGS_ENV", "로컬 환경")

        # 로그 파일 기록
        log_path = os.path.join(OUTPUT_DIR, "consumed_time.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{video_id}]\n")
            f.write(f"시작 시간: {start_datetime}\n")
            f.write(f"방법: {method}\n")
            f.write(f"영상 길이: [{video_length_str}]\n")
            f.write(f"소요 시간: {int(consumed_time // 60)}분 {int(consumed_time % 60)}초\n\n")

        logging.info(f"[📝] 소요 시간 기록 완료: {log_path}")

        logging.info(f"[🎉] 전체 프로세스 완료: {youtube_url}")

    except Exception as e:
        logging.error(f"[❌] 처리 중 오류 발생: {str(e)}")
        raise

def get_video_duration(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps
    except Exception as e:
        logging.error(f"영상 길이 계산 오류: {str(e)}")
        raise

def main():
    logging.info("[👁️] 유튜브 링크 감시 시작...")
    files = os.listdir(INPUT_DIR)
    txt_files = [os.path.join(INPUT_DIR, f) for f in files if f.endswith(".txt")]

    if not txt_files:
        logging.info("[❌] 처리할 링크 파일이 없습니다. 종료합니다.")
        return

    for txt_file in txt_files:
        process_youtube_link(txt_file)

if __name__ == "__main__":
    main()
