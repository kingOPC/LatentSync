"""Render a LatentSync video on two Apple Silicon Macs in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FPS = 25
PRINT_LOCK = threading.Lock()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    duration: float
    video: Path
    audio: Path
    raw_output: Path
    final_output: Path


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log_file=None) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=log_file,
        stderr=subprocess.STDOUT if log_file is not None else None,
    )


def run_streaming(
    command: list[str],
    *,
    segment_index: int,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    child_env = (env or os.environ.copy()).copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    batch = 0
    batches = 1
    current_line = []
    assert process.stdout is not None
    with log_path.open("w") as output_log:
        while True:
            character = process.stdout.read(1)
            if character == "" and process.poll() is not None:
                break
            output_log.write(character)
            if character in {"\r", "\n"}:
                line = ANSI_ESCAPE.sub("", "".join(current_line)).strip()
                current_line.clear()
                if "Doing inference" in line:
                    match = re.search(r"(\d+)\s*/\s*(\d+)", line)
                    if match:
                        batch = int(match.group(1))
                        batches = max(1, int(match.group(2)))
                elif "Sample frames" in line:
                    match = re.search(r"(\d{1,3})%", line)
                    if match:
                        step_fraction = min(100, int(match.group(1))) / 100
                        fraction = min(1.0, (batch + step_fraction) / batches)
                        log(f"DUAL_PROGRESS segment={segment_index} fraction={fraction:.4f}")
            else:
                current_line.append(character)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    log(f"DUAL_PROGRESS segment={segment_index} fraction=1.0000")


def probe_duration(path: Path) -> float:
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
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def ssh(remote: str, command: str, *, log_file=None) -> None:
    run(["ssh", "-o", "BatchMode=yes", remote, command], log_file=log_file)


def worker_env(max_dim: int, steps: int, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LATENTSYNC_MAX_DIM": str(max_dim),
            "LATENTSYNC_NUM_FRAMES": "8",
            "LATENTSYNC_INFERENCE_STEPS": str(steps),
            "LATENTSYNC_SEED": str(seed),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.75",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.65",
            "OMP_NUM_THREADS": "4",
        }
    )
    return env


def render_local(segment_queue: queue.Queue, max_dim: int, steps: int, logs_dir: Path) -> None:
    while True:
        try:
            segment = segment_queue.get_nowait()
        except queue.Empty:
            return
        log(f"[本机] 开始片段 {segment.index + 1}")
        run_streaming(
            [str(ROOT / "run_mac.sh"), str(segment.video), str(segment.audio), str(segment.raw_output)],
            segment_index=segment.index,
            log_path=logs_dir / f"local_{segment.index:04d}.log",
            cwd=ROOT,
            env=worker_env(max_dim, steps, 1247 + segment.index),
        )
        log(f"[本机] 完成片段 {segment.index + 1}")
        segment_queue.task_done()


def render_remote(
    remote: str,
    remote_root: str,
    remote_job: str,
    segment_queue: queue.Queue,
    max_dim: int,
    steps: int,
    logs_dir: Path,
) -> None:
    while True:
        try:
            segment = segment_queue.get_nowait()
        except queue.Empty:
            return
        name = f"segment_{segment.index:04d}"
        remote_video = f"{remote_job}/inputs/{name}.mp4"
        remote_audio = f"{remote_job}/inputs/{name}.wav"
        remote_output = f"{remote_job}/outputs/{name}.mp4"
        variables = {
            "LATENTSYNC_MAX_DIM": str(max_dim),
            "LATENTSYNC_NUM_FRAMES": "8",
            "LATENTSYNC_INFERENCE_STEPS": str(steps),
            "LATENTSYNC_SEED": str(1247 + segment.index),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.75",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.65",
            "OMP_NUM_THREADS": "4",
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        }
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in variables.items())
        command = (
            f"cd {shlex.quote(remote_root)} && {env_text} ./run_mac.sh "
            f"{shlex.quote(remote_video)} {shlex.quote(remote_audio)} {shlex.quote(remote_output)}"
        )
        log(f"[远端] 开始片段 {segment.index + 1}")
        run_streaming(
            ["ssh", "-o", "BatchMode=yes", remote, command],
            segment_index=segment.index,
            log_path=logs_dir / f"remote_{segment.index:04d}.log",
        )
        run(["rsync", "-a", f"{remote}:{remote_output}", str(segment.raw_output)])
        log(f"[远端] 完成片段 {segment.index + 1}")
        segment_queue.task_done()


def normalize_and_split(video: Path, audio: Path, work: Path, max_dim: int, segment_seconds: float) -> list[Segment]:
    audio_duration = probe_duration(audio)
    total_frames = max(1, round(audio_duration * FPS))
    frames_per_segment = max(8, round(segment_seconds * FPS / 8) * 8)
    normalized_video = work / "normalized_video.mp4"
    normalized_audio = work / "normalized_audio.wav"

    log(f"输入音频 {audio_duration:.2f} 秒，共 {total_frames} 帧；每段 {frames_per_segment} 帧")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(video),
            "-t",
            f"{total_frames / FPS:.6f}",
            "-an",
            "-vf",
            f"fps={FPS},scale='if(gt(iw,ih),{max_dim},-2)':'if(gt(iw,ih),-2,{max_dim})':flags=lanczos,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            str(normalized_video),
        ]
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(normalized_audio),
        ]
    )

    inputs = work / "inputs"
    raw = work / "raw"
    finished = work / "finished"
    for directory in (inputs, raw, finished):
        directory.mkdir()

    segments = []
    for index, start_frame in enumerate(range(0, total_frames, frames_per_segment)):
        frame_count = min(frames_per_segment, total_frames - start_frame)
        start = start_frame / FPS
        duration = frame_count / FPS
        stem = f"segment_{index:04d}"
        segment = Segment(
            index=index,
            start=start,
            duration=duration,
            video=inputs / f"{stem}.mp4",
            audio=inputs / f"{stem}.wav",
            raw_output=raw / f"{stem}.mp4",
            final_output=finished / f"{stem}.mp4",
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-t",
                f"{duration:.6f}",
                "-i",
                str(normalized_video),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(segment.video),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-t",
                f"{duration:.6f}",
                "-i",
                str(normalized_audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(segment.audio),
            ]
        )
        segments.append(segment)
    return segments


def trim_and_concat(segments: list[Segment], normalized_audio: Path, output: Path) -> None:
    log("正在校正片段时长并拼接……")
    for segment in segments:
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(segment.raw_output),
                "-t",
                f"{segment.duration:.6f}",
                "-vf",
                f"fps={FPS},format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-an",
                str(segment.final_output),
            ]
        )

    concat_file = segments[0].final_output.parent / "concat.txt"
    concat_file.write_text("".join(f"file '{segment.final_output.as_posix()}'\n" for segment in segments))
    joined_video = segments[0].final_output.parent / "joined_video.mp4"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(joined_video),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    total_duration = sum(segment.duration for segment in segments)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(joined_video),
            "-i",
            str(normalized_audio),
            "-t",
            f"{total_duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LatentSync segments on two Macs in parallel")
    parser.add_argument("video", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--remote", default=os.environ.get("LATENTSYNC_REMOTE"))
    parser.add_argument("--remote-root", default=os.environ.get("LATENTSYNC_REMOTE_ROOT"))
    parser.add_argument("--segment-seconds", type=float, default=4.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-dim", type=int, choices=[1280, 1920], default=1280)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    if not args.remote or not args.remote_root:
        parser.error(
            "two-Mac rendering requires --remote and --remote-root, or the "
            "LATENTSYNC_REMOTE and LATENTSYNC_REMOTE_ROOT environment variables"
        )

    video = args.video.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not video.is_file() or not audio.is_file():
        parser.error("video and audio inputs must exist")
    if args.segment_seconds <= 0 or args.steps <= 0:
        parser.error("segment seconds and inference steps must be positive")

    remote_check = (
        f"test -x {shlex.quote(args.remote_root + '/run_mac.sh')} && "
        f"test -x {shlex.quote(args.remote_root + '/.venv/bin/python')} && "
        "! pgrep -f '[p]ython -m scripts.inference' >/dev/null"
    )
    ssh(args.remote, remote_check)
    if subprocess.run(["pgrep", "-f", "[p]ython -m scripts.inference"], capture_output=True).returncode == 0:
        raise RuntimeError("A local LatentSync inference job is already running")

    output.parent.mkdir(parents=True, exist_ok=True)
    work_parent = output.parent if args.keep_work else None
    work = Path(tempfile.mkdtemp(prefix="latentsync-dual-", dir=work_parent))

    remote_job = f"{args.remote_root}/.dual_jobs/{uuid.uuid4().hex}"
    try:
        segments = normalize_and_split(video, audio, work, args.max_dim, args.segment_seconds)
        logs_dir = work / "logs"
        logs_dir.mkdir()
        log(f"共 {len(segments)} 段：由两台 Mac 按完成速度动态领取")
        log(f"DUAL_TOTAL segments={len(segments)}")

        ssh(args.remote, f"mkdir -p {shlex.quote(remote_job + '/inputs')} {shlex.quote(remote_job + '/outputs')}")
        remote_inputs = [str(path) for segment in segments for path in (segment.video, segment.audio)]
        run(["rsync", "-a", *remote_inputs, f"{args.remote}:{remote_job}/inputs/"])

        segment_queue = queue.Queue()
        for segment in segments:
            segment_queue.put(segment)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(render_local, segment_queue, args.max_dim, args.steps, logs_dir),
                executor.submit(
                    render_remote,
                    args.remote,
                    args.remote_root,
                    remote_job,
                    segment_queue,
                    args.max_dim,
                    args.steps,
                    logs_dir,
                )
            ]
            for future in futures:
                future.result()

        missing = [segment.raw_output for segment in segments if not segment.raw_output.is_file()]
        if missing:
            raise RuntimeError(f"Missing rendered segments: {missing}")
        trim_and_concat(segments, work / "normalized_audio.wav", output)
        log(f"双机生成完成：{output}")
        ssh(
            args.remote,
            "python3 -c " + shlex.quote(f"import shutil; shutil.rmtree({remote_job!r}, ignore_errors=True)"),
        )
        if not args.keep_work:
            shutil.rmtree(work)
    except Exception:
        log(f"任务失败，诊断日志保存在：{work}")
        raise


if __name__ == "__main__":
    main()
