# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict

import cv2

CONFIG_PATH = Path("configs/subtitle_xy.json")


class VideoFrameBrowser:
    def __init__(self, video_path: Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.selected_frame = None

    def on_trackbar(self, val):
        self.current_frame = val
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def browse_frames(self) -> "tuple[bool, Any]":
        cv2.namedWindow("Frame Browser", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", "Frame Browser", 0, self.total_frames - 1, lambda v: self.on_trackbar(v))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow("Frame Browser", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return False, None
            if key == 32:
                self.selected_frame = frame.copy()
                cv2.destroyWindow("Frame Browser")
                return True, self.selected_frame

            self.current_frame = cv2.getTrackbarPos("Frame", "Frame Browser")

        return False, None

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def interactive_roi_selection(video_path: Path) -> tuple:
    """交互式选择字幕区域（保留帧浏览功能）"""
    try:
        # 第一步：浏览视频选择关键帧
        browser = VideoFrameBrowser(video_path)
        success, selected_frame = browser.browse_frames()

        if not success or selected_frame is None:
            return None

        # 第二步：在选定帧上选择ROI
        cv2.namedWindow("Select Subtitle ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select Subtitle ROI", selected_frame, False)
        cv2.destroyAllWindows()

        return roi if all(r > 0 for r in roi) else None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main(video_path: Path):
    series = video_path.parent.name
    config = load_config()

    if series in config:
        print(f"当前配置 {series}: {config[series]}")
        if input("是否覆盖现有配置？(y/n) ").lower() != "y":
            return

    roi = interactive_roi_selection(video_path)
    if not roi:
        print("❌ ROI选择失败")
        return

    config[series] = {"x": int(roi[0]), "y": int(roi[1]), "w": int(roi[2]), "h": int(roi[3])}
    save_config(config)
    print(f"✅ {series} 配置已更新")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="字幕区域配置生成工具")
    parser.add_argument("video_path", type=str, help="输入视频路径")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    main(video_path)
