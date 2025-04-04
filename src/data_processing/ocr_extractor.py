# -*- coding: utf-8 -*-
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

load_dotenv()

BASE_URL = os.environ.get("OCR_OPENAI_BASE_URL")
API_KEY = os.environ.get("OCR_OPENAI_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PPOCR:
    def __init__(self, ocr_config: dict):
        self.ocr_config = ocr_config
        self._init_ocr_engine()
        logging.getLogger("ppocr").setLevel(logging.ERROR)

    def _init_ocr_engine(self):
        use_onnx = self.ocr_config.get("use_onnx", False)
        model_config = self.ocr_config["onnxmodel" if use_onnx else "pdmodel"]
        self.pipeline = PaddleOCR(lang="ch", show_log=False, use_angle_cls=False, use_onnx=use_onnx, **model_config)

    def ocr(self, img: np.ndarray):
        result = self.pipeline.ocr(img)[0]
        return result


class VLMOCR:
    def __init__(self, vlm_config: dict):
        self.vlm_config = vlm_config
        self._init_client()

    def _init_client(self):
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model = self.vlm_config["model"]
        self.prompt = self.vlm_config["prompt"]

    def _encode_image(self, image_array: np.ndarray) -> str:
        img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def ocr(self, img: np.ndarray):
        try:
            image_b64 = self._encode_image(img)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
            )

            text = completion.choices[0].message.content.strip()

            # Format result to match PaddleOCR output format
            # For VLM, we don't have box coordinates, so use the center of the image
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            half_width, half_height = width // 4, height // 4

            # Create a simple bounding box around the center of the image
            box = [
                [center_x - half_width, center_y - half_height],
                [center_x + half_width, center_y - half_height],
                [center_x + half_width, center_y + half_height],
                [center_x - half_width, center_y + half_height],
            ]

            # Return in the same format as PaddleOCR for consistency
            confidence = 0.95  # Assuming high confidence, adjust as needed
            result = [[box, [text, confidence]]]
            return result

        except Exception as e:
            logger.error(f"VLM OCR error: {e}")
            return []


def get_raw_subtitle_filepath(subtitle_dir: Path, video_path: Path) -> Path:
    return subtitle_dir / f"{video_path.stem}.json"


class SubtitleExtractor:
    def __init__(
        self,
        ocr_config_path: str,
        subtitle_config_path: str,
        font_path: str = "models/ocr/fonts/simfang.ttf",
        debug: bool = False,
        debug_dir: Optional[str] = "test/debug",
    ):
        with open(ocr_config_path, "r", encoding="utf-8") as f:
            self.ocr_config = json.load(f)

        self.subtitle_config_path = subtitle_config_path
        self.font_path = font_path
        self.debug = debug
        self.debug_dir = Path(debug_dir)

        self.ocr_engine_type = self.ocr_config["engine"]
        if self.ocr_engine_type == "paddleocr":
            engine_config = self.ocr_config.get("paddleocr")
            self.paddle_ocr = PPOCR(engine_config)

        elif self.ocr_engine_type == "vlm":
            engine_config = self.ocr_config.get("vlm")
            self.vlm_ocr = VLMOCR(engine_config)

        else:
            raise ValueError(f"Invalid OCR engine type: {self.ocr_engine_type}")

        self.subsampling = 5
        self.ssim_threshold = 0.9

    def _get_subtitle_region(self, video_path: str) -> Tuple[int, int, int, int]:
        with open(self.subtitle_config_path, "r", encoding="utf-8") as f:
            self.subtitle_config = json.load(f)

        series = Path(video_path).parent.name
        if series not in self.subtitle_config:
            raise ValueError(f"Series configuration not found: {series}")

        region = self.subtitle_config[series]
        return (region["x"], region["y"], region["w"], region["h"])

    def _preprocess_frame(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        x, y, w, h = region
        cropped = frame[y : y + h, x : x + w]
        return cropped, cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    def _resize_for_ssim(self, image: np.ndarray, scale: float = 0.25) -> np.ndarray:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def image_has_changed(self, img1: Optional[np.ndarray], img2: Optional[np.ndarray]) -> Tuple[bool, float]:
        if img1 is None or img2 is None:
            return True, 0.0

        img1_small = self._resize_for_ssim(img1)
        img2_small = self._resize_for_ssim(img2)
        similarity = structural_similarity(img1_small, img2_small)
        return similarity <= self.ssim_threshold, similarity

    def _ocr_process(self, image: np.ndarray) -> Tuple[List, str, float]:
        if self.ocr_engine_type == "paddleocr":
            return self._paddle_ocr_process(image)
        elif self.ocr_engine_type == "vlm":
            return self._vlm_ocr_process(image)
        else:
            raise ValueError(f"Unknown OCR engine type: {self.ocr_engine_type}")

    def _paddle_ocr_process(self, image: np.ndarray) -> Tuple[List, str, float]:
        result = self.paddle_ocr.ocr(image)
        if not result:
            return [], "", 0.0

        texts, confidences = zip(*[(line[1][0].strip(), line[1][1]) for line in result])
        return result, "".join(texts), sum(confidences) / len(confidences)

    def _vlm_ocr_process(self, image: np.ndarray) -> Tuple[List, str, float]:
        result = self.vlm_ocr.ocr(image)
        if not result:
            return [], "", 0.0

        texts, confidences = zip(*[(line[1][0].strip(), line[1][1]) for line in result])
        return result, "".join(texts), sum(confidences) / len(confidences)

    def _save_debug_info(self, frame: np.ndarray, frame_count: int, result: List, debug_dir: Path) -> None:
        raw_path = debug_dir / f"frame_{frame_count:06d}_raw.png"
        cv2.imwrite(str(raw_path), frame)

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        visualized = draw_ocr(Image.fromarray(frame), boxes, txts, scores, font_path=self.font_path)
        rec_path = debug_dir / f"frame_{frame_count:06d}_rec.png"
        Image.fromarray(visualized).save(rec_path)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_pos: int,
        fps: float,
        region: Tuple[int, int, int, int],
        prev_image: Optional[np.ndarray],
        prev_text: str,
        debug_dir: Optional[Path],
    ) -> Tuple[Optional[np.ndarray], str, Optional[Dict]]:
        cropped, gray_image = self._preprocess_frame(frame, region)
        changed, similarity = self.image_has_changed(gray_image, prev_image)
        if not changed:
            return gray_image, prev_text, None

        result, text, confidence = self._ocr_process(cropped)
        if not text or confidence < 0.8 or text == prev_text:
            return gray_image, prev_text, None

        timestamp = frame_pos / fps
        subtitle = (frame_pos, timestamp, text, confidence, float(similarity))

        if self.debug and debug_dir:
            self._save_debug_info(cropped, frame_pos, result, debug_dir)

        return gray_image, text, subtitle

    def extract(self, video_path: Path, output_dir: Path, subsampling: Optional[int] = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        subsampling = subsampling or self.subsampling
        region = self._get_subtitle_region(video_path)

        video_debug_dir = None
        if self.debug:
            video_debug_dir = self.debug_dir / video_path.stem
            video_debug_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info("Total frames=%d, FPS=%.2f", total_frames, fps)

            subtitles = []
            prev_image, prev_text = None, ""
            progress_bar = tqdm(total=total_frames, unit="frame")

            current_frame_pos = 0
            while current_frame_pos < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = cap.read()
                if not ret:
                    break

                prev_image, prev_text, subtitle = self._process_frame(
                    frame,
                    current_frame_pos,
                    fps,
                    region,
                    prev_image,
                    prev_text,
                    video_debug_dir,
                )
                if subtitle:
                    subtitles.append(subtitle)

                    timeline = self.format_timestamp(subtitle[1])
                    progress_bar.set_description(f"{timeline} {subtitle[2]}")

                progress_bar.update(subsampling)
                current_frame_pos += subsampling

            progress_bar.close()
        finally:
            cap.release()

        output_path = self.get_raw_subtitle_filepath(output_dir, video_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)

        logger.info("Processing completed: %s", video_path.name)
        return output_path

    @staticmethod
    def get_raw_subtitle_filepath(subtitle_dir: Path, video_path: Path) -> Path:
        return subtitle_dir / f"{video_path.stem}.json"

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def main():
    ocr_config_path = Path("configs/ocr_config.json")
    subtitle_config_path = Path("configs/subtitle_xy.json")
    extractor = SubtitleExtractor(ocr_config_path, subtitle_config_path)

    output_subtitle_dir = Path("data/raw/subtitles/json")
    video_playlist_dir = Path("data/raw/videos")

    for videos_dir in video_playlist_dir.iterdir():
        if not videos_dir.is_dir():
            continue

        for video_path in videos_dir.rglob("*.mp4"):
            output_path = extractor.get_raw_subtitle_filepath(output_subtitle_dir, video_path)
            # if output_path.exists():
            #     continue

            try:
                result = extractor.extract(video_path, output_subtitle_dir)
                logger.info("Subtitle file saved to: %s", result)
            except Exception as e:
                logger.error("Processing failed: %s", video_path, exc_info=e)


if __name__ == "__main__":
    main()
