# 📄 deploy_kegs/train/finetune.py - Whisper 모델을 4비트 양자화로 파인튜닝하는 스크립트

import os
import glob
import sys
import json
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, Audio
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# GPU 가용성 확인
print(torch.cuda.is_available())

# 설정 로드
with open("train/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["base_model"]
OUTPUT_DIR = config["output_dir"]
DATASET_PATH = config["dataset_path"]
SAMPLE_RATE = 16000

def load_model_and_processor():
    # 1) 4비트 양자화 로드 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    # 2) 모델 로드
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=False
    )
    # 3) k-bit 양자화 학습 준비
    model = prepare_model_for_kbit_training(model)

    # 4) Processor 로드
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language="ko",
        task="transcribe",
        local_files_only=False
    )
    # 5) LoRA 어댑터 설정
    peft_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.eval()
    return model, processor

# def load_model_and_processor():
#     processor = WhisperProcessor.from_pretrained(MODEL_NAME)
#     model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
#     model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
#     return model, processor

def load_combined_dataset():
    tsv_files = [str(p) for p in Path(DATASET_PATH).glob("*.tsv")]
    if not tsv_files:
        print("[경고] 학습용 .tsv 데이터가 존재하지 않습니다. 빈 모델만 저장합니다.")
        return None

    datasets = []
    for tsv_file in tsv_files:
        ds = load_dataset("csv", data_files={"train": tsv_file}, delimiter="\t")["train"]
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    combined = combined.map(lambda x: {**x, "audio": x["audio"].replace("\\", "/")}, desc="경로 슬래시 정규화")
    combined = combined.cast_column("audio", Audio(sampling_rate=16000))
    return combined

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features[0]
    labels = processor.tokenizer(
        batch["text"], padding=True, truncation=True
        ).input_ids
    batch["labels"] = labels
    return batch

class DataCollatorWhisper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        batch = {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels": torch.tensor(self.processor.tokenizer.pad(
                {"input_ids": label_features}, padding=True, return_tensors="pt"
            )["input_ids"], dtype=torch.long)
        }
        return batch

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, processor = load_model_and_processor()

    dataset = load_combined_dataset()
    if dataset is None:
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print("[완료] 빈 모델만 저장되었습니다.")
        return

    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset.column_names,
        batched=False
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-4,
        logging_dir="./logs",
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_32bit",
		gradient_checkpointing=True,
        report_to=None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        data_collator=DataCollatorWhisper(processor),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[완료] 모델 파인튜닝 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
