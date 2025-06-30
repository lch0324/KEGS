# 📄 deploy_kegs/train/generate_dataset.py - YouTube 오디오 다운로드 및 자막 생성

import os
import sys
import subprocess
from urllib.parse import urlparse, parse_qs

import torch
import torchaudio
from moviepy import AudioFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Silero VAD 로드
vad_model, vad_utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', force_reload=False
)
get_speech_timestamps = vad_utils[0]

# 경로 설정
DATASET_DIR = "./dataset"
MODEL_DIR = "./models/whisper-large-v3-turbo-finetuned"
SAMPLE_RATE = 16000
MIN_VALID_TEXT_LEN = 5


def clean_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return f"https://www.youtube.com/watch?v={qs['v'][0]}"
    return url.split('&')[0]


def get_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip('/')
    qs = parse_qs(parsed.query)
    return qs.get('v', ['unknown'])[0]


def download_audio_from_youtube(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'video.mp4')
    wav_path = os.path.join(output_dir, 'audio.wav')

    subprocess.run([
        'yt-dlp', '-f', 'bestaudio', '-o', video_path, url
    ], check=True)

    clip = AudioFileClip(video_path)
    clip.write_audiofile(
        wav_path,
        fps=SAMPLE_RATE,
        codec='pcm_s16le',
        ffmpeg_params=['-ac', '1']
    )
    clip.close()

    try:
        os.remove(video_path)
    except OSError:
        pass

    return wav_path


def generate_subtitles(audio_path: str, processor, model) -> list[dict]:
    """
    Hugging Face pipeline을 사용해 전체 오디오에 대해 한국어 자막 세그먼트 리스트 반환
    """
    from transformers import pipeline

    # device 설정
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor,
        device=device,
        chunk_length_s=30,
        stride_length_s=(5, 5),
        return_timestamps="sentence",
        generate_kwargs={"language": "ko", "task": "transcribe"}
    )
    # 파일 경로 대신 array 입력 가능, pipeline이 파일 읽음
    result = asr(audio_path)
    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        if text and len(text) >= MIN_VALID_TEXT_LEN:
            segments.append({"start": float(start), "end": float(end), "text": text})
    return segments


def save_to_tsv(segments: list[dict], audio_path: str, tsv_path: str):
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    with open(tsv_path, 'w', encoding='utf-8') as f:
        # Header with newline
        f.write('audio\ttext\n')
        for seg in segments:
            # Each line ends with newline
            f.write(f"{audio_path}\t{seg['text']}\n")


def main():
    if len(sys.argv) != 2:
        print('Usage: python train/generate_dataset.py [YouTube URL]')
        sys.exit(1)

    url = clean_youtube_url(sys.argv[1])
    vid = get_video_id(url)
    audio_dir = os.path.join(DATASET_DIR, 'audio', vid)
    tsv_path = os.path.join(DATASET_DIR, f'{vid}.tsv')

    audio_path = download_audio_from_youtube(url, audio_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = WhisperProcessor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device)
    # ⚠️ Remove forced_decoder_ids to avoid conflicts with decoder_input_ids
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    segments = generate_subtitles(audio_path, processor, model)
    save_to_tsv(segments, audio_path, tsv_path)
    print(f'[✅] Dataset TSV saved to: {tsv_path}')


if __name__ == '__main__':
    main()
