# -*- coding: utf-8 -*-
import hashlib
import json
import subprocess
from pathlib import Path

import srt
from pydub import AudioSegment
from tqdm import tqdm

SAMPLE_RATE = 16000


def extract_audio(video_path: Path, audio_path: Path):
    if not audio_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-q:a",
                "0",
                "-map",
                "a",
                "-ar",
                str(SAMPLE_RATE),
                audio_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def split_audio_by_srt(audio_path: Path, srt_path: Path, clip_dir: Path):
    audio = AudioSegment.from_file(audio_path)
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f))

    video_id = clip_dir.parts[-1]
    clip_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for sub in tqdm(subtitles, desc="Processing clips"):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)

        clip_filename = f"{video_id}_{start_ms:07d}_{end_ms:07d}.wav"
        clip_path = clip_dir / clip_filename

        clip = audio[start_ms:end_ms]
        clip.export(
            clip_path,
            format="wav",
            parameters=["-ac", "1", "-ar", str(SAMPLE_RATE)],
        )

        data.append(
            {
                "audio": {
                    "path": str(clip_path),
                    "sampling_rate": SAMPLE_RATE,
                },
                "text": sub.content.strip(),
                "source": str(audio_path),
            }
        )

    return data


def main():
    video_dir = Path("data/raw/videos")
    srt_dir = Path("data/raw/subtitles/srt")
    audio_dir = Path("data/raw/audios")
    output_dir = Path("data/dataset")

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for subdir in Path(video_dir).iterdir():
        if not subdir.is_dir():
            continue

    i = 0
    for video_file in subdir.rglob("*.mp4"):
        video_name = video_file.stem
        srt_path = srt_dir / f"{video_file.stem}.srt"

        if not srt_path.exists():
            print(f"Warning: SRT not found for {video_file}")
            continue

        audio_path = audio_dir / f"{video_name}.wav"
        extract_audio(video_file, audio_path)

        video_id = hashlib.md5(audio_path.stem.encode()).hexdigest()[:5]
        clip_dir = output_dir / "clips" / video_id
        if clip_dir.exists():
            print(f"Warning: Skip {video_name}")
            continue

        i += 1
        data = split_audio_by_srt(audio_path, srt_path, clip_dir)
        all_data.extend(data)

        if i >= 3:
            break

    with open(output_dir / "metadata.jsonl", "a", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
