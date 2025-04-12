# -*- coding: utf-8 -*-
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,  # noqa: F401
)
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not existed: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(dataset_config: dict, processor: WhisperProcessor) -> Dataset:
    def prepare_features(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = load_dataset(dataset_config["raw_parquet_cache_dir"], cache_dir=dataset_config["raw_arrow_cache_dir"])
    processed_dataset = dataset["train"].map(prepare_features, num_proc=dataset_config["num_proc"], remove_columns=["audio"])

    return processed_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
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


def train(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # ------------------------------------------------------------------ #
    # --------------------------- Model  ------------------------------- #
    # ------------------------------------------------------------------ #
    model_config = config["model"]

    model_dir = model_config["model_dir"]
    language = model_config["language"]
    task = model_config["task"]

    logger.info(f"Loading model: {model_dir}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_dir, language=language, task=task)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    if config["training"].get("gradient_checkpointing", False):
        model.config.use_cache = False

    # ------------------------------------------------------------------ #
    # --------------------------- Dataset ------------------------------ #
    # ------------------------------------------------------------------ #
    dataset_config = config["dataset"]
    # dataset = prepare_dataset(dataset_config, processor)
    # dataset.save_to_disk(dataset_config["map_cache_dir"], num_proc=8)
    dataset = load_from_disk(dataset_config["map_cache_dir"])

    train_test_split = dataset.train_test_split(test_size=dataset_config["test_size"])
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(eval_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    logger.info("Dataset preparation completed")
    # ------------------------------------------------------------------ #
    # --------------------------- Mertics ------------------------------ #
    # ------------------------------------------------------------------ #
    metric = evaluate.load("src/metrics/wer")

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
    training_config = config["training"]
    pprint(training_config)

    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        training_config["fp16"] = False

    training_args = Seq2SeqTrainingArguments(
        report_to=["tensorboard"],
        disable_tqdm=True,
        logging_steps=100,
        **training_config,
    )

    last_checkpoint = None
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(output_dir)
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
    config_path = "configs/whisper-small_config.json"
    train(config_path)


if __name__ == "__main__":
    main()
