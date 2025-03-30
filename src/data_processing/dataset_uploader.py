# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from datasets import Audio, Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class DatasetUploader:
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        repo_id: str,
        private: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.repo_id = repo_id
        self.private = private
        self.metadata_path = self.dataset_dir / "metadata.jsonl"
        self.clips_dir = self.dataset_dir / "clips"

        self._validate_dataset_structure()
        self.api = HfApi()

    def _validate_dataset_structure(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.jsonl file does not exist: {self.metadata_path}")

        if not self.clips_dir.exists() or not self.clips_dir.is_dir():
            raise FileNotFoundError(f"clips directory does not exist: {self.clips_dir}")

        logger.info(f"Dataset directory structure validation passed: {self.dataset_dir}")

    def _read_metadata(self) -> List[Dict]:
        metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))

        logger.info(f"Successfully read {len(metadata)} metadata records")
        return metadata

    def _prepare_dataset(self) -> Dataset:
        metadata = self._read_metadata()
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

    def upload(self) -> None:
        dataset = self._prepare_dataset()
        logger.info(f"Starting to upload dataset to {self.repo_id}")
        dataset.push_to_hub(self.repo_id, token=self.token, private=self.private)
        logger.info(f"Dataset successfully uploaded to {self.repo_id}")
        logger.info(f"You can view your dataset at https://huggingface.co/datasets/{self.repo_id}")


def main():
    dataset_dir = "data/dataset"
    repo_id = "i18nJack/fuzhouhua"
    private = False
    try:
        uploader = DatasetUploader(dataset_dir=dataset_dir, repo_id=repo_id, private=private)
        uploader.upload()
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
