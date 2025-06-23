# train/finetune.py

import os
import glob
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets, Audio
import json
import torch

print(torch.cuda.is_available())

# 설정 로드
with open("train/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["base_model"]
OUTPUT_DIR = config["output_dir"]
DATASET_PATH = config["dataset_path"]

def load_model_and_processor():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    return model, processor

def load_combined_dataset():
    tsv_files = glob.glob(os.path.join(DATASET_PATH, "*.tsv"))
    if not tsv_files:
        print("[경고] 학습용 .tsv 데이터가 존재하지 않습니다. 빈 모델만 저장합니다.")
        return None

    datasets = []
    for tsv_file in tsv_files:
        ds = load_dataset("csv", data_files={"train": tsv_file}, delimiter="\t")["train"]
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    combined = combined.cast_column("audio", Audio(sampling_rate=16000))
    return combined

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000).input_features[0]
    labels = processor.tokenizer(batch["text"], padding=True, truncation=True).input_ids
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

    dataset = dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-5,
        logging_dir="./logs",
        save_steps=200,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,  # FutureWarning 대응
        data_collator=DataCollatorWhisper(processor),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[완료] 모델 파인튜닝 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
