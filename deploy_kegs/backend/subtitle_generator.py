# 📄 deploy_kegs/backend/subtitle_generator.py - Whisper로부터 SRT 자막 생성

import os
import re
import torch
import torchaudio
from moviepy import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from konlpy.tag import Komoran
import logging
import config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Silero VAD 모델 로드
vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
get_speech_timestamps = utils[0]

SAMPLE_RATE = 16000
MIN_VALID_TEXT_LEN = 5

def calculate_char_positions(sentence, morphs):
    positions = []
    p = 0
    for morph in morphs:
        while p < len(sentence) and sentence[p] == ' ':
            p += 1
        start = p
        for c in morph:
            if p < len(sentence) and sentence[p] == c:
                p += 1
            else:
                break
        positions.append(start)
    return positions

def find_morph_end_position(sentence, start_pos, morph):
    p = start_pos
    m = 0
    while p < len(sentence) and m < len(morph):
        if sentence[p] == morph[m]:
            m += 1
        p += 1
    return p

def improved_semantic_split(text, min_len=12, max_len=80):
    os.makedirs("./logs", exist_ok=True)

    komoran = Komoran()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    subtitle_chunks = []

    for sentence in sentences:
        tokens = komoran.pos(sentence)
        morphs = [morph for morph, _ in tokens]
        morph_positions = calculate_char_positions(sentence, morphs)

        split_indices = set()

        for idx, (word, pos) in enumerate(tokens):
            if pos == 'EF':
                ef_end_pos = find_morph_end_position(sentence, morph_positions[idx], word)
                if ef_end_pos < len(sentence) and sentence[ef_end_pos] == ' ':
                    split_indices.add(ef_end_pos)
            elif pos == 'MAJ' or (word == ',' and pos == 'SP'):
                split_indices.add(morph_positions[idx] + len(word))

        for ec_idx, (morph, pos) in enumerate(tokens):
            if pos == 'EC':
                has_vv = any(
                    ec_idx - hop >= 0 and tokens[ec_idx - hop][1].startswith('VV')
                    for hop in range(1, 4)
                )
                if has_vv:
                    ec_end_pos = find_morph_end_position(sentence, morph_positions[ec_idx], morph)
                    if ec_end_pos < len(sentence) and sentence[ec_end_pos] == ' ':
                        split_indices.add(ec_end_pos)

        split_indices = sorted(split_indices)

        chunks = []
        last = 0
        for idx in split_indices:
            chunk = sentence[last:idx].strip()
            if chunk:
                chunks.append(chunk)
            last = idx
        if last < len(sentence):
            chunk = sentence[last:].strip()
            if chunk:
                chunks.append(chunk)

        # 짧은 절 병합
        merged_chunks = []
        if len(chunks) == 1:
            merged_chunks.append(chunks[0])
        else:
            i = 0
            while i < len(chunks):
                if len(chunks[i]) >= min_len:
                    merged_chunks.append(chunks[i])
                    i += 1
                else:
                    if i == 0:
                        merged_chunk = chunks[i] + " " + chunks[i + 1]
                        merged_chunks.append(merged_chunk.strip())
                        i += 2
                    elif i == len(chunks) - 1:
                        merged_chunks[-1] += " " + chunks[i]
                        i += 1
                    else:
                        merged_chunk = chunks[i] + " " + chunks[i + 1]
                        merged_chunks.append(merged_chunk.strip())
                        i += 2

        subtitle_chunks.extend(merged_chunks)

    final_subtitles = []
    for chunk in subtitle_chunks:
        if len(chunk) > max_len:
            tokens = komoran.pos(chunk)
            morphs = [word for word, _ in tokens]
            split_points = [idx for idx, (word, pos) in enumerate(tokens) if pos in ['SP', 'MAJ']]

            char_positions = []
            char_idx = 0
            for idx, word in enumerate(morphs):
                char_positions.append(char_idx)
                char_idx += len(word)
                if idx < len(morphs) - 1:
                    char_idx += 1

            candidate_chars = [char_positions[i] for i in split_points if i < len(char_positions)]
            split_point = None
            if candidate_chars:
                mid = len(chunk) // 2
                split_point = min(candidate_chars, key=lambda x: abs(x - mid))
            else:
                spaces = [m.start() for m in re.finditer(r'\s', chunk)]
                if spaces:
                    split_point = min(spaces, key=lambda x: abs(x - len(chunk) // 2))

            if split_point:
                final_subtitles.append(chunk[:split_point + 1].strip())
                final_subtitles.append(chunk[split_point + 1:].strip())
            else:
                final_subtitles.append(chunk.strip())
        else:
            final_subtitles.append(chunk.strip())

    return final_subtitles

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt_from_video(video_path, srt_output_path, model_path=config.MODEL_PATH):
    try:
        logging.info("[🎬] 영상에서 오디오 추출 중...")
        video_path = os.path.abspath(video_path)
        audio_path = video_path.replace(".mp4", ".wav")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, fps=SAMPLE_RATE, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])
        clip.close()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 추출 실패: {audio_path}")

        logging.info("[🧠] Whisper 모델 로딩 중...")
        processor = WhisperProcessor.from_pretrained(model_path, language="ko", task="transcribe", local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.eval()

        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.squeeze(0)

        logging.info("[🔍] 음성 구간 감지 중...")
        speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=SAMPLE_RATE)
        logging.info(f"[✅] 감지된 음성 구간 수: {len(speech_timestamps)}")

        segments = []
        segment_index = 1

        for ts in speech_timestamps:
            start_sec = ts['start'] / SAMPLE_RATE
            end_sec = ts['end'] / SAMPLE_RATE
            chunk = waveform[ts['start']:ts['end']]
            duration = end_sec - start_sec
            input_features = processor.feature_extractor(
                chunk.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_features=input_features.to(model.device),
                    return_dict_in_generate=True,
                    return_timestamps=True,
                    language="ko",  # 한국어 강제 지정
                     task="transcribe",
                    max_length=448,
                    suppress_tokens=[]
                )

            # segments와 decoded 텍스트
            batch_segs = outputs["segments"]
            if not batch_segs or not batch_segs[0]:
                continue  # segments 없으면 다음 chunk로
            chunk_segs = batch_segs[0]  # 첫 배치
            decoded_list = processor.batch_decode(outputs["sequences"], skip_special_tokens=True)

            for seg, txt in zip(chunk_segs, decoded_list):
                decoded = txt.strip()
                logging.info(f"[🔤] Whisper 출력: {decoded}")
                if not decoded or len(decoded) < MIN_VALID_TEXT_LEN:
                    continue

            logging.info("[✂️] 자막 문장 분할 중...")
            lines = improved_semantic_split(decoded, min_len=12, max_len=80)
            logging.info(f"[📌] 분할된 자막 수: {len(lines)}")

            total_chars = sum(len(line) for line in lines)
            local_time = start_sec

            for line in lines:
                proportion = len(line) / total_chars
                line_duration = proportion * duration
                segments.append({
                    "index": segment_index,
                    "start": local_time,
                    "end": local_time + line_duration,
                    "text": line
                })
                segment_index += 1
                local_time += line_duration

        if not segments:
            raise ValueError("자막 생성 실패: segment가 비어있음")

        with open(srt_output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"{seg['index']}\n")
                f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")

        if not os.path.exists(srt_output_path):
            raise FileNotFoundError(f"SRT 파일 생성 실패: {srt_output_path}")

        logging.info(f"[✅] SRT 파일 저장 완료: {srt_output_path}")
    except Exception as e:
        logging.error(f"[❌] 자막 생성 중 오류 발생: {str(e)}")
        raise
