import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import gdown
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DOWNLOAD_ROOT = Path(tempfile.gettempdir()) / "video_merger_downloads"
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Video Merger API")

DOWNLOADS: Dict[str, Path] = {}


class MergeRequest(BaseModel):
    drive_link: str = Field(..., description="Public Google Drive folder URL")
    music_url: Optional[str] = Field(None, description="Optional public MP3 URL")


class MergeResponse(BaseModel):
    download_url: str


def run_ffmpeg(args: List[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{result.stderr}")


def run_ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe duration failed for {path}: {result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid duration for {path}: {result.stdout}") from exc


def has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def find_video_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()]
    return sorted(files)


def download_music(music_url: str, out_path: Path) -> None:
    with requests.get(music_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_thumbnail(video_path: Path, out_path: Path) -> None:
    duration = max(run_ffprobe_duration(video_path), 0.1)
    snap_ts = min(max(duration * 0.2, 0.1), duration - 0.05 if duration > 0.2 else duration)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{snap_ts:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def analyze_with_gemini(video_paths: List[Path], temp_dir: Path) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "ordered_clips": [
                {
                    "filename": path.name,
                    "start_sec": 0.0,
                    "end_sec": max(0.1, run_ffprobe_duration(path)),
                }
                for path in video_paths
            ]
        }

    model = genai.GenerativeModel("gemini-2.0-flash")
    parts: List[Any] = [
        (
            "You are preparing clips for a merged video. "
            "Return strict JSON only in this exact schema: "
            '{"ordered_clips":[{"filename":"string","start_sec":0.0,"end_sec":1.0}]}. '
            "Order clips for best narrative flow and suggest trim points. "
            "Do not include unknown filenames."
        )
    ]

    for clip in video_paths:
        duration = run_ffprobe_duration(clip)
        thumb = temp_dir / f"{clip.stem}_thumb.jpg"
        extract_thumbnail(clip, thumb)
        with thumb.open("rb") as image_file:
            image_bytes = image_file.read()
        parts.append(f"Clip filename={clip.name}, duration_sec={duration:.2f}")
        parts.append({"mime_type": "image/jpeg", "data": image_bytes})

    response = model.generate_content(parts)
    try:
        parsed = parse_json_response(response.text or "")
    except Exception as exc:
        raise RuntimeError(f"Gemini JSON parse failed. Raw response: {response.text}") from exc

    if "ordered_clips" not in parsed or not isinstance(parsed["ordered_clips"], list):
        raise RuntimeError(f"Unexpected Gemini output: {parsed}")

    return parsed


def normalize_plan(plan: Dict[str, Any], video_paths: List[Path]) -> List[Dict[str, Any]]:
    by_name = {p.name: p for p in video_paths}
    normalized: List[Dict[str, Any]] = []
    seen = set()

    for item in plan.get("ordered_clips", []):
        filename = item.get("filename")
        if filename not in by_name or filename in seen:
            continue
        clip_path = by_name[filename]
        duration = run_ffprobe_duration(clip_path)
        start = float(item.get("start_sec", 0.0))
        end = float(item.get("end_sec", duration))
        start = max(0.0, min(start, duration - 0.05 if duration > 0.05 else duration))
        end = max(start + 0.05, min(end, duration))
        normalized.append({"path": clip_path, "start": start, "end": end})
        seen.add(filename)

    for clip in video_paths:
        if clip.name not in seen:
            duration = run_ffprobe_duration(clip)
            normalized.append({"path": clip, "start": 0.0, "end": duration})

    return normalized


def trim_clip(in_path: Path, out_path: Path, start: float, end: float) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out_path),
        ]
    )


def concat_clips(input_paths: List[Path], out_path: Path, work_dir: Path) -> None:
    concat_file = work_dir / "concat.txt"
    lines = [f"file '{path.as_posix()}'" for path in input_paths]
    concat_file.write_text("\n".join(lines), encoding="utf-8")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(out_path),
        ]
    )


def add_background_music(video_path: Path, music_path: Path, out_path: Path) -> None:
    if has_audio_stream(video_path):
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(music_path),
                "-filter_complex",
                "[1:a]volume=0.25[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_path),
            ]
        )
    else:
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(music_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_path),
            ]
        )


def process_merge(drive_link: str, music_url: Optional[str]) -> Path:
    with tempfile.TemporaryDirectory(prefix="video_merge_") as temp_str:
        temp_dir = Path(temp_str)
        source_dir = temp_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        downloaded = gdown.download_folder(
            url=drive_link,
            output=str(source_dir),
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
        )
        if downloaded is None:
            raise RuntimeError("Failed to download Google Drive folder.")

        video_files = find_video_files(source_dir)
        if not video_files:
            raise RuntimeError("No video files found in provided Drive folder.")

        gemini_plan = analyze_with_gemini(video_files, temp_dir)
        processing_plan = normalize_plan(gemini_plan, video_files)

        cut_dir = temp_dir / "cut"
        cut_dir.mkdir(parents=True, exist_ok=True)

        cut_files: List[Path] = []
        for idx, item in enumerate(processing_plan):
            cut_out = cut_dir / f"clip_{idx:04d}.mp4"
            trim_clip(item["path"], cut_out, item["start"], item["end"])
            cut_files.append(cut_out)

        merged_video = temp_dir / "merged.mp4"
        concat_clips(cut_files, merged_video, temp_dir)

        final_video = temp_dir / "final.mp4"
        if music_url:
            music_file = temp_dir / "music.mp3"
            download_music(music_url, music_file)
            add_background_music(merged_video, music_file, final_video)
        else:
            shutil.copy2(merged_video, final_video)

        download_id = str(uuid.uuid4())
        persisted_dir = DOWNLOAD_ROOT / download_id
        persisted_dir.mkdir(parents=True, exist_ok=True)
        persisted_path = persisted_dir / "final.mp4"
        shutil.copy2(final_video, persisted_path)
        return persisted_path


def cleanup_download_file(download_id: str) -> None:
    file_path = DOWNLOADS.pop(download_id, None)
    if not file_path:
        return
    parent = file_path.parent
    try:
        if file_path.exists():
            file_path.unlink()
        if parent.exists():
            parent.rmdir()
    except Exception:
        pass


@app.post("/merge", response_model=MergeResponse)
def merge_videos(payload: MergeRequest, request: Request) -> MergeResponse:
    try:
        final_path = process_merge(payload.drive_link, payload.music_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    download_id = final_path.parent.name
    DOWNLOADS[download_id] = final_path
    download_url = str(request.base_url).rstrip("/") + f"/download/{download_id}"
    return MergeResponse(download_url=download_url)


@app.get("/download/{download_id}")
def download_result(download_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    path = DOWNLOADS.get(download_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Download file not found or expired.")

    background_tasks.add_task(cleanup_download_file, download_id)
    return FileResponse(path, media_type="video/mp4", filename="merged_video.mp4")
