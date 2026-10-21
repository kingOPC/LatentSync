"""OpenCV/FFmpeg media helpers that work on macOS arm64."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable

import cv2
import numpy as np


def _open_video(video_path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def get_video_frame_count(video_path: str) -> int:
    capture = _open_video(video_path)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return count


def get_video_fps(video_path: str) -> float:
    capture = _open_video(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"Could not determine video FPS: {video_path}")
    return fps


def read_video_frames(video_path: str, indices: Iterable[int] | None = None) -> np.ndarray:
    """Read RGB video frames, optionally at specific zero-based indices."""
    capture = _open_video(video_path)
    frames: list[np.ndarray] = []
    try:
        if indices is None:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            for index in [int(value) for value in indices]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise IndexError(f"Could not read frame {index} from {video_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return np.stack(frames, axis=0)


def read_audio_ffmpeg(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode any FFmpeg-supported audio source to mono float32 samples."""
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(audio_path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
        "-ar", str(sample_rate), "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise RuntimeError(f"Audio contains no readable samples: {audio_path}")
    return samples
