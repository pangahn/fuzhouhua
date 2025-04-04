# -*- coding: utf-8 -*-
import sys
from datetime import timedelta
from pathlib import Path

import srt
from tqdm import tqdm

# ruff: noqa: E402
SCRIPT_DIR = Path(__file__).resolve()
PROJECR_IR = "/" + "/".join(SCRIPT_DIR.parts[1 : SCRIPT_DIR.parts.index("src")])
sys.path.insert(0, str(PROJECR_IR + "/src"))

from data_processing.srt_generator import SentenceTransformerSimilarity


class SRTChecker:
    def __init__(
        self,
        time_gap_threshold=0.3,
        semantic_threshold=0.7,
        min_duration=0.5,
        sbert_model_name="all-MiniLM-L6-v2",
        use_onnx=True,
    ):
        self.time_gap_threshold = time_gap_threshold
        self.semantic_threshold = semantic_threshold
        self.min_duration = min_duration

        self.model = SentenceTransformerSimilarity(sbert_model_name, use_onnx)

    @staticmethod
    def _substring_similarity(text1, text2):
        if len(text1) >= 3 and text1 in text2:
            return 0.8

    def calculate_similarity(self, text1, text2):
        bert_similarity = self.model.calculate_similarity(text1, text2)

        if self._substring_similarity(text1[:3], text2) or self._substring_similarity(text2[:3], text1):
            return max(0.8, bert_similarity) if bert_similarity > 0.3 else bert_similarity
        return bert_similarity

    @staticmethod
    def create_subtitle(index, start_time, end_time, content):
        if end_time <= start_time:
            end_time = start_time + timedelta(milliseconds=100)
        return srt.Subtitle(index=index, start=start_time, end=end_time, content=content)

    def _merge_semantic_subtitles(self, subtitles):
        if len(subtitles) <= 1:
            return subtitles

        i = 0
        while i < len(subtitles) - 1:
            current = subtitles[i]
            next_sub = subtitles[i + 1]

            time_gap = (next_sub.start - current.end).total_seconds()
            if time_gap <= self.time_gap_threshold:
                similarity = self.calculate_similarity(current.content, next_sub.content)
                if similarity >= self.semantic_threshold:
                    content = current.content
                    if len(next_sub.content) > len(current.content):
                        content = next_sub.content
                    merged_sub = self.create_subtitle(current.index, current.start, next_sub.end, content)

                    subtitles[i] = merged_sub
                    subtitles.pop(i + 1)
                    continue

            i += 1

        return subtitles

    def process_srt_file(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            subtitles = list(srt.parse(f.read()))

        subtitles = self._merge_semantic_subtitles(subtitles)
        subtitles[-1].end = subtitles[-1].start + timedelta(milliseconds=100)

        for i, subtitle in enumerate(subtitles, 1):
            subtitle.index = i

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))


def main():
    output_path = Path("data/raw/subtitles/checked")
    output_path.mkdir(exist_ok=True)

    checker = SRTChecker(
        time_gap_threshold=0.5,
        semantic_threshold=0.8,
        sbert_model_name="distiluse-base-multilingual-cased-v2",
        use_onnx=True,
    )

    input_srt_folder = Path("data/raw/subtitles/srt")
    subtitle_list = list(input_srt_folder.rglob("*.srt"))
    for subtitle_file in tqdm(subtitle_list, desc="Processing subtitles", unit="file"):
        output_file = output_path / (subtitle_file.stem + ".srt")
        if output_file.exists():
            continue
        checker.process_srt_file(subtitle_file, output_file)


if __name__ == "__main__":
    main()
