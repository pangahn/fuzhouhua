# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, Union

import torch
import torchaudio
from datasets import load_dataset
from tabulate import tabulate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_config(config_path: Union[str, Path]) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not existed: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_audio(audio_data, target_sample_rate=16000):
    waveform = torch.tensor(audio_data["array"]).unsqueeze(0)  # (1, samples)
    sampling_rate = audio_data["sampling_rate"]

    if sampling_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sample_rate)
        waveform = resampler(waveform)

    return waveform.squeeze(), target_sample_rate


def transcribe_audio(example, model, processor):
    audio = example["audio"]
    waveform, sample_rate = preprocess_audio(audio)

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    example["prediction"] = transcription
    return example


def evaluate_model(dataset, ckpt_path, pretrained_model_path, num_samples=5):
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_path)
    processor = WhisperProcessor.from_pretrained(pretrained_model_path)

    results = dataset["train"].select(range(num_samples)).map(lambda example: transcribe_audio(example, model, processor))

    table_data = []
    for i in range(num_samples):
        table_data.append([i + 1, results[i]["text"], results[i]["prediction"]])

    headers = ["index", "text", "prediction"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    return results


def main():
    config_path = "configs/whisper-large-v3_config.json"
    config = load_config(config_path)
    pretrained_model_path = config["model"]["model_dir"]

    dataset_config = config["dataset"]
    dataset = load_dataset(dataset_config["raw_parquet_cache_dir"], cache_dir=dataset_config["raw_arrow_cache_dir"])

    output_dir = Path(config["training"]["output_dir"])
    checkpoint_id = "checkpoint-10000"
    ckpt_path = output_dir / checkpoint_id
    print(f"Using checkpoint from: {ckpt_path}")

    evaluate_model(dataset, ckpt_path, pretrained_model_path, num_samples=5)


if __name__ == "__main__":
    main()
