"""OpenCV/FFmpeg replacement for the Decord-backed AVReader."""

from __future__ import annotations

import numpy as np

from .media import get_video_fps, read_audio_ffmpeg, read_video_frames


class AVReader:
    """Index synchronized audio and RGB frames without requiring Decord."""

    def __init__(self, uri, ctx=None, sample_rate=44100, mono=True, width=-1, height=-1, num_threads=0, fault_tol=-1):
        del ctx, mono, width, height, num_threads, fault_tol
        self.video_frames = read_video_frames(str(uri))
        self.audio_samples = read_audio_ffmpeg(str(uri), sample_rate)
        self.sample_rate = sample_rate
        self.fps = get_video_fps(str(uri))

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self))))
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for {len(self)} frames")
        audio_start = round(idx / self.fps * self.sample_rate)
        audio_end = round((idx + 1) / self.fps * self.sample_rate)
        return self.audio_samples[audio_start:audio_end], self.video_frames[idx]

    def get_batch(self, indices):
        indices = np.asarray(list(indices), dtype=np.int64)
        indices[indices < 0] += len(self)
        if not ((0 <= indices) & (indices < len(self))).all():
            raise IndexError("Frame index out of bounds")
        audio = [self[index][0] for index in indices]
        return audio, self.video_frames[indices]
