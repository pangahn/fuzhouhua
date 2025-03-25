# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

from pytubefix import Playlist
from pytubefix.cli import on_progress
from tqdm import tqdm


def load_existing_metadata(file_path):
    metadata = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                metadata[data["video_id"]] = data
    return metadata


def save_metadata(file_path, metadata):
    with open(file_path, "w", encoding="utf-8") as f:
        for data in metadata.values():
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def download_playlist(url, base_folder, metadata_file, playlist_name=None):
    pl = Playlist(url)
    playlist_name = pl.title.replace(" ", "_") if playlist_name is None else playlist_name
    playlist_folder = os.path.join(base_folder, f"yt_playlist_{playlist_name}")
    os.makedirs(playlist_folder, exist_ok=True)

    print(f"\n开始下载播放列表: {pl.title}")
    print(f"共有 {len(pl.video_urls)} 个视频")

    existing_metadata = load_existing_metadata(metadata_file)

    for video in tqdm(pl.videos, desc="下载进度", unit="视频"):
        video.register_on_progress_callback(on_progress)
        tqdm.write(f"处理视频: {video.title}")
        ys = video.streams.get_highest_resolution()
        video_path = ys.download(output_path=playlist_folder, skip_existing=True)
        video_meta = {
            "source": "youtube",
            "video_id": video.video_id,
            "video_url": video.watch_url,
            "playlist_url": url,
            "title": video.title,
            "duration": video.length,
            "file_path": video_path,
        }
        existing_metadata[video.video_id] = video_meta

    save_metadata(metadata_file, existing_metadata)
    print(f"播放列表 '{pl.title}' 下载完成")


def main():
    base_folder = Path("data/raw/videos")
    base_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = base_folder / "videometa.jsonl"

    with open("configs/playlist.json", "r", encoding="utf-8") as f:
        playlist_urls = json.load(f)

    for playlist_name, url in playlist_urls.items():
        download_playlist(url, base_folder, metadata_file, playlist_name=playlist_name)

    print("\n所有播放列表下载完成, 元信息已保存到videometa.jsonl文件中")


if __name__ == "__main__":
    main()
