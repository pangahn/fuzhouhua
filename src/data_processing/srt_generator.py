# -*- coding: utf-8 -*-
import json
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import srt
from sentence_transformers import SentenceTransformer
from srt import Subtitle
from tqdm import tqdm


class SentenceTransformerSimilarity:
    def __init__(self, model_name: str, use_onnx: bool = True):
        os.environ["ORT_PROVIDERS"] = "CPUExecutionProvider"
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"

        if use_onnx:
            self.model = SentenceTransformer(
                model_name,
                backend="onnx",
                model_kwargs={
                    "file_name": "onnx/model.onnx",
                    "provider": "CPUExecutionProvider",
                },
            )
        else:
            self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        similarity = self.model.similarity(embeddings[0], embeddings[1])
        return float(similarity[0][0])


class SRTGenerator:
    def __init__(
        self,
        confidence_threshold=0.7,
        frame_similarity_threshold=0.8,
        text_similarity_threshold=0.8,
        max_gap=1.5,
        frequency_threshold=2,
        end_time_offset=0.1,
        min_duration=0.5,
        sbert_model_name="all-MiniLM-L6-v2",
        use_onnx=True,
    ):
        self.confidence_threshold = confidence_threshold
        self.frame_similarity_threshold = frame_similarity_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.max_gap = max_gap
        self.frequency_threshold = frequency_threshold
        self.end_time_offset = end_time_offset
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
        return Subtitle(index=index, start=start_time, end=end_time, content=content)

    def aggregate(self, input_file, output_file):
        subtitles = []
        current_group = []

        with open(input_file, "r", encoding="utf-8") as f:
            lines = json.load(f)

        for line in lines:
            if len(line) == 4:
                seconds, text, confidence, frame_similarity = line
            elif len(line) == 5:
                _, seconds, text, confidence, frame_similarity = line
            else:
                continue

            if float(confidence) < self.confidence_threshold:
                continue

            time = timedelta(seconds=seconds)

            if current_group:
                last_time, last_text, _ = current_group[-1]
                time_diff = (time - last_time).total_seconds()
                text_similarity = self.calculate_similarity(text, last_text)
                frame_similarity = float(frame_similarity)

                # 添加到当前字幕组
                is_group_text = True if text_similarity >= self.text_similarity_threshold else False
                is_group_frame = True if frame_similarity >= self.frame_similarity_threshold else False
                is_group_timeline = True if time_diff <= self.max_gap else False

                if (is_group_text or is_group_frame) and is_group_timeline:
                    current_group.append((time, text, float(confidence)))
                    continue

            # 当前字幕组分组完成, 开始合并该字幕组
            if current_group:
                self._process_group(current_group, subtitles)

            # 开始新的字幕组
            current_group = [(time, text, float(confidence))]

        # 处理最后一组字幕
        if current_group:
            self._process_group(current_group, subtitles)

        # 后处理优化
        self._merge_adjacent_subtitles(subtitles)
        self._adjust_subtitle_times(subtitles)

        # 保存字幕文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))

    def _process_group(self, group: List[Tuple], subtitles: List[Subtitle]):
        if not group:
            return

        texts = [item[1] for item in group]
        counter = Counter(texts)

        ((most_common, count),) = counter.most_common(1) or [(None, 0)]

        if count >= self.frequency_threshold:
            selected = most_common
        else:
            max_length = max(len(t) for t in texts)
            candidates = [item for item in group if len(item[1]) == max_length]
            selected = max(candidates, key=lambda x: x[2])[1]

        start_time = group[0][0]
        end_time = group[-1][0]
        subtitles.append(
            self.create_subtitle(
                len(subtitles) + 1,
                start_time,
                end_time,
                self.post_text_process(selected),
            )
        )

    @staticmethod
    def post_text_process(srt_content):
        srt_content = srt_content.replace("但语", "俚语")
        return srt_content

    def _merge_adjacent_subtitles(self, subtitles):
        merged = []
        if not subtitles:
            return

        current = subtitles[0]
        for sub in subtitles[1:]:
            gap = (sub.start - current.end).total_seconds()
            if current.content == sub.content and gap <= self.max_gap:
                current = self.create_subtitle(current.index, current.start, sub.end, current.content)
            else:
                merged.append(current)
                current = sub
        merged.append(current)

        subtitles[:] = merged

    def _adjust_subtitle_times(self, subtitles):
        min_dur = timedelta(seconds=self.min_duration)

        for i in range(len(subtitles) - 1):
            new_end = subtitles[i + 1].start - timedelta(seconds=self.end_time_offset)
            subtitles[i].end = max(new_end, subtitles[i].start + min_dur)

        if subtitles:
            last = subtitles[-1]
            if last.end - last.start < min_dur:
                last.end = last.start + min_dur


def main():
    srt_folder = Path("data/raw/subtitles/srt")
    srt_folder.mkdir(parents=True, exist_ok=True)

    generator = SRTGenerator(
        confidence_threshold=0.7,
        frame_similarity_threshold=0.75,
        text_similarity_threshold=0.8,
        max_gap=10,
        min_duration=0.3,
        sbert_model_name="distiluse-base-multilingual-cased-v2",
        use_onnx=True,
    )

    raw_subtitle_dir = Path("data/raw/subtitles/json")
    raw_subtitle_list = list(raw_subtitle_dir.rglob("*.json"))
    for subtitle_file in tqdm(raw_subtitle_list, desc="Processing subtitles", unit="file"):
        output_file = srt_folder / (subtitle_file.stem + ".srt")
        generator.aggregate(subtitle_file, output_file)


if __name__ == "__main__":
    main()
