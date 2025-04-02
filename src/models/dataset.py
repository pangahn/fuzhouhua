# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Union

from datasets import Audio, Dataset
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


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
    dataset_dir = "data/dataset"
    repo_id = "i18nJack/fuzhouhua"
    private = False

    dataset = build_dataset(dataset_dir)

    try:
        dataset.push_to_hub(repo_id, private=private)
        logger.info(f"You can view your dataset at https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
