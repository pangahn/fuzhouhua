# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PPOCR:
    def __init__(self, ocr_config_path: str):
        self._load_config(ocr_config_path)
        self._init_ocr_engine()
        logging.getLogger("ppocr").setLevel(logging.ERROR)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            self.ocr_config = json.load(f)

    def _init_ocr_engine(self):
        use_onnx = self.ocr_config.get("use_onnx", False)
        model_config = self.ocr_config["paddleocr"]["onnxmodel" if use_onnx else "pdmodel"]
        self.pipeline = PaddleOCR(lang="ch", show_log=False, use_angle_cls=False, use_onnx=use_onnx, **model_config)

    def ocr(self, img: np.ndarray):
        result = self.pipeline.ocr(img)[0]
        return result


def get_raw_subtitle_filepath(subtitle_dir: Path, video_path: Path) -> Path:
    return subtitle_dir / f"{video_path.stem}.json"


class SubtitleExtractor:
    def __init__(
        self,
        ocr_config_path: str = "configs/ocr_config.json",
        subtitle_config_path: str = "configs/subtitle_xy.json",
        font_path: str = "models/ocr/fonts/simfang.ttf",
        debug: bool = False,
        debug_dir: Optional[str] = "test/debug",
    ):
        self.subsampling = 5
        self.ssim_threshold = 0.9

        self.font_path = font_path
        self.subtitle_config_path = subtitle_config_path
        self.ocr_engine = PPOCR(ocr_config_path)

        self.debug = debug
        self.debug_dir = Path(debug_dir)

    def _get_subtitle_region(self, video_path: str) -> Tuple[int, int, int, int]:
        with open(self.subtitle_config_path, "r", encoding="utf-8") as f:
            self.subtitle_config = json.load(f)

        series = Path(video_path).parent.name
        if series not in self.subtitle_config:
            raise ValueError(f"未找到系列配置: {series}")

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
        result = self.ocr_engine.ocr(image)
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
        """处理单个视频帧并返回识别结果"""
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
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info("视频信息: 总帧数=%d, FPS=%.2f", total_frames, fps)

            subtitles = []
            prev_image, prev_text = None, ""
            progress_bar = tqdm(total=total_frames, desc="处理进度", unit="frame")

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

                progress_bar.update(subsampling)
                current_frame_pos += subsampling

            progress_bar.close()
        finally:
            cap.release()

        output_path = get_raw_subtitle_filepath(output_dir, video_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)

        logger.info("处理完成: %s", video_path.name)
        return output_path


if __name__ == "__main__":
    extractor = SubtitleExtractor(debug=False)
    subtitle_dir = Path("data/raw/subtitles/json")
    playlist_dir = Path("data/raw/videos")

    for videos_dir in Path(playlist_dir).iterdir():
        if not videos_dir.is_dir():
            continue

        for video_path in videos_dir.rglob("*.mp4"):
            output_path = get_raw_subtitle_filepath(subtitle_dir, video_path)
            if output_path.exists():
                continue

            try:
                result = extractor.extract(video_path, subtitle_dir)
                logger.info("字幕文件已保存至: %s", result)
            except Exception as e:
                logger.error("处理失败: %s", video_path, exc_info=e)
