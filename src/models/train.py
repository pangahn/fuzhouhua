# -*- coding: utf-8 -*-
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import evaluate
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not existed: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(dataset: Dataset, processor: WhisperProcessor) -> Dataset:
    def prepare_features(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processed_dataset = dataset.map(prepare_features, remove_columns=["audio"])
    return processed_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def train(config_path, dataset_dir: Path):
    config = load_config(config_path)
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_name_or_path = config.get("model_name", "openai/whisper-large-v3")
    language = config.get("language", "zh")
    task = config.get("task", "transcribe")
    logger.info(f"Loading model: {model_name_or_path}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    # ------------------------------------------------------------------ #
    # --------------------------- Dataset ------------------------------ #
    # ------------------------------------------------------------------ #
    dataset_config = config["dataset"]

    if dataset_dir.exists():
        base_dataset = load_from_disk(dataset_dir)
    else:
        base_dataset = load_dataset(dataset_config["repo_id"])

    dataset = prepare_dataset(base_dataset, processor)
    train_test_split = dataset.train_test_split(test_size=dataset_config["test_size"])
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(eval_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    # ------------------------------------------------------------------ #
    # --------------------------- Evaluate ----------------------------- #
    # ------------------------------------------------------------------ #
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # ------------------------------------------------------------------ #
    # --------------------------- Training ----------------------------- #
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        config["training"]["fp16"] = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        report_to=["tensorboard"],
        **config.get("training"),
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Found checkpoint: {last_checkpoint}, will resume training from here")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main():
    dataset_dir = "data/dataset"
    config_path = "configs/whisper_config.json"
    train(config_path=config_path, dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
