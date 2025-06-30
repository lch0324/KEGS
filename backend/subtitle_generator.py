# 📄 backend/subtitle_generator.py - Whisper로부터 SRT 자막 생성

import os
import re
import torch
import torchaudio
from moviepy import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from konlpy.tag import Komoran
import uuid
import config

vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
get_speech_timestamps = utils[0]

SAMPLE_RATE = 16000
MIN_VALID_TEXT_LEN = 5

def calculate_char_positions(sentence, morphs):
    """문장에서 형태소 단어들이 시작하는 위치 계산"""
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
    """Whisper 원문에서 morph의 끝나는 실제 위치를 찾음"""
    p = start_pos
    m = 0  # morph의 인덱스

    while p < len(sentence) and m < len(morph):
        if sentence[p] == morph[m]:
            m += 1
        p += 1

    return p  # 끝 위치 (exclusive)

def improved_semantic_split(text, video_id, min_len=12, max_len=80):
    """
    긴 Whisper 자막 텍스트를 자연스럽게 의미 단위로 분할하고, 디버그 로그 저장.

    Args:
        text (str): Whisper로부터 생성된 텍스트.
        video_id (str): 디버그 로그 파일명에 사용할 ID.
        min_len (int): 하나의 자막 최소 길이 (기본 12자).
        max_len (int): 하나의 자막 최대 길이 (기본 80자).
    
    Returns:
        list: 의미 단위로 나눈 자막 리스트.
    """
    os.makedirs("../logs", exist_ok=True)
    log_path = f"../logs/split_debug_{video_id}.txt"
    log_file = open(log_path, "a", encoding="utf-8")

    komoran = Komoran()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    subtitle_chunks = []

    log_file.write(f"✅ [0] 원본 텍스트:\n{text}\n\n")
    log_file.write(f"✅ [1] 문장 부호 기준 1차 분할 결과:\n")
    for s in sentences:
        log_file.write(f"- {s}\n")
    log_file.write("\n")

    for sentence in sentences:
        tokens = komoran.pos(sentence)
        morphs = [morph for morph, _ in tokens]
        morph_positions = calculate_char_positions(sentence, morphs)

        split_indices = set()
        ec_candidates = []

        for idx, (word, pos) in enumerate(tokens):
            if pos == 'EF':
                end_pos = morph_positions[idx] + len(word)
                if end_pos < len(sentence) and sentence[end_pos] == ' ':
                    split_indices.add(end_pos)
            elif pos == 'MAJ' or (word == ',' and pos == 'SP'):
                split_indices.add(morph_positions[idx] + len(word))

        for ec_idx, (morph, pos) in enumerate(tokens):
            if pos == 'EC':
                has_vv = any(
                    ec_idx - hop >= 0 and tokens[ec_idx - hop][1].startswith('VV')
                    for hop in range(1, 4)
                )
                if has_vv:
                    ec_start_pos = morph_positions[ec_idx]
                    ec_end_pos = find_morph_end_position(sentence, ec_start_pos, morph)

                    ec_candidates.append((ec_idx, ec_end_pos))

                    # EC 끝난 바로 다음이 공백이면 분리
                    if ec_end_pos < len(sentence) and sentence[ec_end_pos] == ' ':
                        split_indices.add(ec_end_pos)

        split_indices = sorted(split_indices)

        log_file.write(f"🎯 문장: {sentence}\n")
        log_file.write(f"✅ [2] 형태소 분석 결과:\n{tokens}\n\n")
        log_file.write(f"✅ [3] EC 후보 인덱스:\n")
        for ec_idx, ec_end_pos in ec_candidates:
            log_file.write(f"- EC index: {ec_idx}, EC end pos: {ec_end_pos}\n")
        log_file.write("\n")
        log_file.write(f"✅ [4] split_indices:\n{split_indices}\n\n")

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

        log_file.write(f"✅ [5] 초벌 분할 결과:\n")
        for c in chunks:
            log_file.write(f"- {c}\n")
        log_file.write("\n")

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

        log_file.write(f"✅ [6] 짧은 절 병합 결과:\n")
        for m in merged_chunks:
            log_file.write(f"- {m}\n")
        log_file.write("\n")

        subtitle_chunks.extend(merged_chunks)

    final_subtitles = []
    for chunk in subtitle_chunks:
        if len(chunk) > max_len:
            tokens = komoran.pos(chunk)
            morphs = [word for word, _ in tokens]

            split_points = []
            for idx, (word, pos) in enumerate(tokens):
                if pos == 'SP' and word == ',':
                    split_points.append(idx)
                if pos == 'MAJ':
                    split_points.append(idx)

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
                first = chunk[:split_point + 1].strip()
                second = chunk[split_point + 1:].strip()
                final_subtitles.append(first)
                final_subtitles.append(second)
            else:
                final_subtitles.append(chunk.strip())
        else:
            final_subtitles.append(chunk.strip())

    log_file.write(f"✅ [7] 최종 결과:\n")
    for m in final_subtitles:
        log_file.write(f"- {m}\n")
    log_file.write("\n====================\n\n")
    log_file.close()
    print(f"[🪵 디버그] 분리 결과 로그 저장 완료 → '{log_path}'")

    return final_subtitles

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt_from_video(video_path, srt_output_path, model_path=config.MODEL_PATH):
    print("[🎬] 영상에서 오디오 추출 중...")
    audio_path = video_path.replace(".mp4", ".wav")

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, fps=SAMPLE_RATE, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])
    clip.close()

    print("[🧠] Whisper 모델 로딩 중...")
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.squeeze(0)

    print("[🔍] 음성 구간 감지 중...")
    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=SAMPLE_RATE)
    print(f"[✅] 감지된 음성 구간 수: {len(speech_timestamps)}")

    segments = []
    segment_index = 1
    video_id = str(uuid.uuid4())[:8]

    # raw_transcript.txt 덮어쓰기
    os.makedirs("../logs", exist_ok=True)
    with open("../logs/raw_transcript.txt", "w", encoding="utf-8") as tf:
        tf.write("")  # 초기화

    for ts in speech_timestamps:
        start_sec = ts['start'] / SAMPLE_RATE
        end_sec = ts['end'] / SAMPLE_RATE
        chunk = waveform[ts['start']:ts['end']]
        duration = end_sec - start_sec

        input_features = processor.feature_extractor(
            chunk.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features

        # decoder_input_ids = torch.tensor([processor.get_decoder_prompt_ids(language="ko", task="transcribe")], device=model.device)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=input_features,
                # decoder_input_ids=decoder_input_ids,
                max_length=448,
                suppress_tokens=[]
            )

        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if not decoded or len(decoded) < MIN_VALID_TEXT_LEN:
            continue

        # Whisper 원문 저장
        with open("../logs/raw_transcript.txt", "a", encoding="utf-8") as tf:
            tf.write(decoded + "\n\n")

        # Improved split
        lines = improved_semantic_split(decoded, video_id, min_len=12, max_len=80)

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

    print(f"[✅] 최종 자막 줄 수: {len(segments)}")

    with open(srt_output_path, "w", encoding="utf-8-sig") as f:
        for seg in segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")