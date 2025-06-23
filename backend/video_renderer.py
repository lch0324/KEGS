import os
import subprocess
import shutil

def render_video_with_subtitles(youtube_url, srt_path, output_dir, download_only=False):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 유튜브 영상 다운로드 (mp4)
    video_path = os.path.join(output_dir, "original_video.mp4")
    if not os.path.exists(video_path):
        command = [
            "yt-dlp",
            "-f", "best[ext=mp4][vcodec!*=av01]",
            "-o", video_path,
            youtube_url
        ]
        subprocess.run(command, check=True)

    if download_only:
        return video_path

    # 2. srt 자막 복사
    target_srt_path = os.path.join(output_dir, "subtitles.srt")
    if os.path.abspath(srt_path) != os.path.abspath(target_srt_path):
        shutil.copy2(srt_path, target_srt_path)

    # 3. ffmpeg로 자막 입힌 영상 생성 (스타일 적용)
    rel_video = os.path.relpath(video_path, start=os.getcwd()).replace("\\", "/")
    rel_srt = os.path.relpath(target_srt_path, start=os.getcwd()).replace("\\", "/")
    output_video_path = os.path.join(output_dir, "subtitled_output.mp4")
    rel_output = os.path.relpath(output_video_path, start=os.getcwd()).replace("\\", "/")

    # ✅ ffmpeg subtitles 필터에 force_style 옵션 추가
    subtitle_filter = f"subtitles='{rel_srt}':force_style='FontName=Noto Sans KR,FontSize=20,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=3,OutlineColour=&H00000000,Outline=1'"

    command = [
        "ffmpeg", "-y",
        "-i", rel_video,
        "-vf", subtitle_filter,
        "-c:a", "copy",
        rel_output
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("[ffmpeg 오류 출력]:")
        print(e.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg 자막 입히기 실패. stderr 로그 참조.")

    return os.path.abspath(output_video_path)
