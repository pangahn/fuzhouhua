# -*- coding: utf-8 -*-
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import srt
from Levenshtein import ratio
from srt import Subtitle
from tqdm import tqdm


class SRTGenerator:
    def __init__(
        self,
        confidence_threshold=0.7,
        similarity_threshold=0.8,
        max_gap=1.5,
        frequency_threshold=2,
        end_time_offset=0.1,
        min_duration=0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.max_gap = max_gap
        self.frequency_threshold = frequency_threshold
        self.end_time_offset = end_time_offset
        self.min_duration = min_duration

    @staticmethod
    def text_similarity(text1, text2):
        return ratio(text1, text2)

    @staticmethod
    def create_subtitle(index, start_time, end_time, content):
        if end_time <= start_time:
            end_time = start_time + timedelta(milliseconds=100)
        return Subtitle(index=index, start=start_time, end=end_time, content=content)

    def aggregate_subtitles(self, input_file, output_file):
        subtitles = []
        current_group = []

        with open(input_file, "r", encoding="utf-8") as f:
            lines = json.load(f)

        for line in lines:
            if len(line) == 4:
                seconds, text, confidence, _ = line
            elif len(line) == 5:
                _, seconds, text, confidence, _ = line
            else:
                continue

            if float(confidence) < self.confidence_threshold:
                continue

            time = timedelta(seconds=seconds)

            if current_group:
                last_time, last_text, _ = current_group[-1]
                time_diff = (time - last_time).total_seconds()
                similarity = self.text_similarity(text, last_text)

                # 添加到当前字幕组
                if similarity >= self.similarity_threshold and time_diff <= self.max_gap:
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
        subtitles.append(self.create_subtitle(len(subtitles) + 1, start_time, end_time, selected))

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


if __name__ == "__main__":
    srt_folder = Path("data/raw/subtitles/srt")
    raw_subtitle_dir = Path("data/raw/subtitles/json")

    generator = SRTGenerator(
        confidence_threshold=0.7,
        similarity_threshold=0.8,
        max_gap=3,
        min_duration=0.8,
    )

    for subtitle_file in tqdm(Path(raw_subtitle_dir).rglob("*.json"), desc="Processing subtitles"):
        output_file = Path(srt_folder) / (subtitle_file.stem + ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        generator.aggregate_subtitles(subtitle_file, output_file)
