# -*- coding: utf-8 -*-
import json
import logging
import os
from pathlib import Path
from typing import Union

from datasets import Audio, Dataset
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
assert "HF_TOKEN" in os.environ, "HF_TOKEN not found in environment variables"


def build_dataset(dataset_dir: Union[str, Path]) -> Dataset:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.jsonl"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl file does not exist: {metadata_path}")

    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    logger.info(f"Successfully read {len(metadata)} metadata records")

    processed_data = []
    for item in metadata:
        audio_path = item["audio"]["path"]
        if not Path(audio_path).exists():
            logger.warning(f"Audio file does not exist, skipping: {audio_path}")
            continue

        processed_item = {
            "audio": str(audio_path),
            "text": item["text"],
            "source": item.get("source", ""),
        }
        processed_data.append(processed_item)

    logger.info(f"Successfully processed {len(processed_data)} data items")

    dataset = Dataset.from_list(processed_data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def main():
    input_clip_dir = "data/clips"
    output_dataset_path = "data/dataset"
    data_version = "1.0.0"
    commit_message = "Upload dataset"

    dataset = build_dataset(input_clip_dir)
    dataset.save_to_disk(output_dataset_path)
    logger.info(f"Dataset saved to {output_dataset_path}")

    config_path = "configs/whisper_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    push_to_hub = config["dataset"]["push_to_hub"]
    repo_id = config["dataset"]["repo_id"]
    private = config["dataset"]["private"]

    if push_to_hub:
        dataset.push_to_hub(repo_id, private=private, data_dir=data_version, commit_message=commit_message)
        logger.info(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
