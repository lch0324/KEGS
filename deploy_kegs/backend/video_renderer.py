import os
import subprocess
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def render_video_with_subtitles(video_path, srt_path, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # srt 파일 복사
        target_srt_path = os.path.splitext(video_path)[0] + ".srt"
        if os.path.abspath(srt_path) != os.path.abspath(target_srt_path):
            shutil.copy2(srt_path, target_srt_path)

        abs_video = os.path.abspath(video_path).replace("\\", "/")
        abs_srt = os.path.abspath(target_srt_path).replace("\\", "/")
        abs_output = os.path.abspath(output_path).replace("\\", "/")

        # font_path = "/home/lch0324/.fonts/NotoSansKR-Regular.ttf"

        subtitle_filter = f"subtitles={abs_srt}"

        ffmpeg_path = "/data/lch0324/downloads/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"

        command = [
            ffmpeg_path, "-y",
            "-i", abs_video,
            "-vf", subtitle_filter,
            "-c:a", "copy",
            abs_output
        ]

        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"자막 입힌 mp4 파일 생성 실패: {output_path}")

        logging.info(f"[✅] 자막 입힌 영상 생성 완료: {output_path}")

    except subprocess.CalledProcessError as e:
        logging.error("[❌] ffmpeg 실행 오류 발생")
        logging.error(e.stderr.decode("utf-8", errors="ignore"))
        raise
    except Exception as e:
        logging.error(f"[❌] 자막 입히기 중 오류 발생: {str(e)}")
        raise
