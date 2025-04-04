# -*- coding: utf-8 -*-
import hashlib
import subprocess
from pathlib import Path

import jsonlines
import srt
from pydub import AudioSegment
from tqdm import tqdm

SAMPLE_RATE = 16000


def get_video_files(video_dir):
    video_files = []
    for subdir in Path(video_dir).iterdir():
        if not subdir.is_dir():
            continue
        video_files.extend(list(subdir.rglob("*.mp4")))
    return video_files


def extract_audio(video_path, audio_path):
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


def create_clip(audio, start_ms, end_ms, clip_path):
    clip = audio[start_ms:end_ms]
    clip.export(
        clip_path,
        format="wav",
        parameters=["-ac", "1", "-ar", str(SAMPLE_RATE)],
    )


def process_single_video(video_file, srt_dir, audio_dir, output_dir):
    video_name = video_file.stem
    srt_path = srt_dir / f"{video_name}.srt"

    if not srt_path.exists():
        print(f"SRT not found for {video_file}")
        return []

    # Read subtitle file
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f))

    # Extract audio
    audio_path = audio_dir / f"{video_name}.wav"
    extract_audio(video_file, audio_path)

    # Create clip directory
    video_id = hashlib.md5(audio_path.stem.encode()).hexdigest()[:5]
    clip_dir = output_dir / "clips" / video_id

    # Check if already processed
    if clip_dir.exists():
        wav_files = list(clip_dir.glob("*.wav"))
        if len(wav_files) >= len(subtitles) - 10:
            return []

    clip_dir.mkdir(parents=True, exist_ok=True)

    # Process audio segments
    audio = AudioSegment.from_file(audio_path)
    clips_data = []

    for sub in tqdm(subtitles[:-1], desc=f"Processing: {video_name}", leave=False):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)

        clip_filename = f"{video_id}_{start_ms:07d}_{end_ms:07d}.wav"
        clip_path = clip_dir / clip_filename

        create_clip(audio, start_ms, end_ms, clip_path)

        clips_data.append(
            {
                "audio": {
                    "path": str(clip_path),
                    "sampling_rate": SAMPLE_RATE,
                },
                "text": sub.content.strip(),
                "source": str(audio_path),
            }
        )

    # Save clip metadata
    metadata_path = clip_dir / "metadata.jsonl"
    with jsonlines.open(metadata_path, mode="w") as writer:
        writer.write_all(clips_data)

    return clips_data


def merge_metadata(output_dir):
    all_metadata = []
    clips_dir = output_dir / "clips"

    for clip_dir in tqdm(list(clips_dir.iterdir()), desc="Merging metadata files"):
        metadata_path = clip_dir / "metadata.jsonl"
        if metadata_path.exists():
            with jsonlines.open(metadata_path, "r") as reader:
                all_metadata.extend(list(reader))

    final_metadata_path = output_dir / "metadata.jsonl"
    with jsonlines.open(final_metadata_path, mode="w") as writer:
        writer.write_all(all_metadata)


def main():
    video_dir = Path("data/raw/videos")
    srt_dir = Path("data/raw/subtitles/checked")
    audio_dir = Path("data/raw/audios")
    output_dir = Path("data/dataset")

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = get_video_files(video_dir)
    for video_file in tqdm(video_files, desc="Processing videos", position=0, leave=True):
        process_single_video(video_file, srt_dir, audio_dir, output_dir)

    merge_metadata(output_dir)


if __name__ == "__main__":
    main()
