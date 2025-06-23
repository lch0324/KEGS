# train/generate_dataset.py

import os
import sys
import whisper
from pydub import AudioSegment
import subprocess
from urllib.parse import urlparse, parse_qs

DATASET_DIR = "./dataset"
MODEL_DIR = "./models/whisper-large-v3-finetuned"

def clean_youtube_url(url):
    """유튜브 URL을 정제하여 오류 방지"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "v" in query:
        return f"https://www.youtube.com/watch?v={query['v'][0]}"
    return url.split("&")[0]

def get_video_id(url):
    """유튜브 영상 ID 추출"""
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    query = parse_qs(parsed.query)
    return query["v"][0] if "v" in query else "unknown"

def download_audio_from_youtube(url: str, audio_output_path: str):
    try:
        mp3_path = os.path.join(audio_output_path, "temp_audio.mp3")
        command = [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", mp3_path,
            url
        ]
        subprocess.run(command, check=True)

        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        final_path = os.path.join(audio_output_path, "audio.wav")
        audio.export(final_path, format="wav")
        os.remove(mp3_path)

        return final_path

    except Exception as e:
        print(f"[오류] 유튜브 오디오 다운로드 실패: {e}")
        sys.exit(1)

def generate_subtitles(audio_path: str, model):
    try:
        result = model.transcribe(audio_path, language="ko")
        return result["segments"]
    except Exception as e:
        print(f"[오류] 자막 생성 실패: {e}")
        sys.exit(1)

def save_to_tsv(segments, audio_path, tsv_path):
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("audio\ttext\n")
        for seg in segments:
            f.write(f"{audio_path}\t{seg['text'].strip()}\n")

def main():
    if len(sys.argv) != 2:
        print("사용법: python train/generate_dataset.py [유튜브 링크]")
        sys.exit(1)

    url = clean_youtube_url(sys.argv[1])
    video_id = get_video_id(url)
    audio_output_path = os.path.join(DATASET_DIR, "audio", video_id)
    tsv_path = os.path.join(DATASET_DIR, f"{video_id}.tsv")

    os.makedirs(audio_output_path, exist_ok=True)

    audio_path = download_audio_from_youtube(url, audio_output_path)

    model = whisper.load_model("large-v3")
    segments = generate_subtitles(audio_path, model)

    save_to_tsv(segments, audio_path, tsv_path)
    print(f"[완료] 데이터셋 저장 경로: {tsv_path}")

if __name__ == "__main__":
    main()
