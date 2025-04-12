# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import numpy as np
from paddleocr import draw_ocr
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_processing.ocr_extractor import PPOCR


def run_ocr(ocr_engine, image_path):
    rgb_img = Image.open(image_path).convert("RGB")
    return ocr_engine.ocr(np.array(rgb_img)), rgb_img


def save_result_image(image, result, font_path, output_path):
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    Image.fromarray(im_show).save(output_path)


def main():
    ocr_config_path = "configs/ocr_config.json"
    with open(ocr_config_path, "r", encoding="utf-8") as f:
        ocr_config = json.load(f)

    engine_config = ocr_config["paddleocr"]
    paddle_ocr = PPOCR(engine_config)

    img_path = "tests/imgs/test_ppocr.png"
    result, rgb_img = run_ocr(paddle_ocr, img_path)

    for line in result:
        print(f"Text: {line[1][0]}, Confidence: {line[1][1]:.4f}")

    output_path = "tests/imgs/ppocr_result.jpg"
    save_result_image(rgb_img, result, ocr_config["font_path"], output_path)
    print(f"Result image saved to {output_path}")


if __name__ == "__main__":
    main()
