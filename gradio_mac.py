"""Local, memory-safe Gradio interface for Apple Silicon Macs."""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
RUN_SCRIPT = ROOT_DIR / "run_mac.sh"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TTS_VOICES = {
    "婷婷（普通话女声）": "Tingting",
    "Flo（普通话女声）": "Flo (中文（中国大陆）)",
    "Reed（普通话男声）": "Reed (中文（中国大陆）)",
    "Eddy（普通话男声）": "Eddy (中文（中国大陆）)",
}


def _synthesize_text(text: str, voice_label: str, speech_rate: int, work_dir: Path) -> Path:
    source_text = work_dir / "speech.txt"
    source_aiff = work_dir / "speech.aiff"
    output_wav = work_dir / "speech.wav"
    source_text.write_text(text.strip(), encoding="utf-8")

    voice = TTS_VOICES.get(voice_label, "Tingting")
    subprocess.run(
        ["say", "-v", voice, "-r", str(int(speech_rate)), "-f", str(source_text), "-o", str(source_aiff)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_aiff),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_wav


def preview_voice(text: str, voice_label: str, speech_rate: int) -> str:
    if not (text or "").strip():
        raise gr.Error("请先输入需要试听的配音文案。")

    preview_dir = OUTPUT_DIR / "tts_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    preview = preview_dir / f"preview_{timestamp}.wav"
    try:
        with tempfile.TemporaryDirectory(prefix=".preview-", dir=preview_dir) as temporary:
            synthesized = _synthesize_text(text, voice_label, speech_rate, Path(temporary))
            shutil.copy2(synthesized, preview)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or str(error)).strip()
        raise gr.Error(f"配音试听生成失败：{details}") from error

    previews = sorted(preview_dir.glob("preview_*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_preview in previews[20:]:
        stale_preview.unlink(missing_ok=True)
    return str(preview)


def _update_progress_from_log(line: str, state: dict, progress) -> None:
    line = ANSI_ESCAPE.sub("", line)
    if not line:
        return

    total_match = re.match(r"DUAL_TOTAL segments=(\d+)", line)
    if total_match:
        state["dual_total"] = max(1, int(total_match.group(1)))
        state["dual_segments"] = {}
        progress(0.10, desc=f"双机任务已拆成 {state['dual_total']} 个片段……")
        return
    segment_match = re.match(r"DUAL_PROGRESS segment=(\d+) fraction=([0-9.]+)", line)
    if segment_match:
        segment = int(segment_match.group(1))
        fraction = min(1.0, float(segment_match.group(2)))
        state.setdefault("dual_segments", {})[segment] = fraction
        total = state.get("dual_total", 1)
        overall = sum(state["dual_segments"].values()) / total
        progress(0.10 + 0.85 * overall, desc=f"两台 Mac 正在并行生成：{overall:.0%}")
        return

    if "[1/2]" in line:
        progress(0.02, desc="正在转换为 25fps 标准视频……")
    elif "[2/2]" in line:
        progress(0.08, desc="正在加载 MPS 模型与人脸检测器……")
    elif "Affine transforming" in line:
        state["phase"] = "align"
        progress(0.15, desc="正在检测并对齐人脸……")
    elif "Doing inference" in line:
        state["phase"] = "inference"
        match = re.search(r"(\d+)\s*/\s*(\d+)", line)
        if match:
            state["batch"] = int(match.group(1))
            state["batches"] = max(1, int(match.group(2)))
    elif "Restoring" in line:
        state["phase"] = "restore"
        progress(0.92, desc="正在把生成的嘴部区域回贴到原视频……")
    elif line.startswith("Done:"):
        progress(0.99, desc="正在检查并载入输出视频……")

    percent_match = re.search(r"(\d{1,3})%", line)
    if not percent_match:
        return
    percent = min(100, int(percent_match.group(1))) / 100
    if state.get("phase") == "align":
        progress(0.15 + 0.05 * percent, desc=f"正在检测并对齐人脸：{percent:.0%}")
    elif "Sample frames" in line:
        batch = state.get("batch", 0)
        batches = state.get("batches", 1)
        overall = (batch + percent) / batches
        progress(0.20 + 0.70 * overall, desc=f"正在进行 20 步口型生成：{overall:.0%}")
    elif state.get("phase") == "restore":
        progress(0.92 + 0.07 * percent, desc=f"正在回贴并合成视频：{percent:.0%}")


def generate_video(
    video_path: str,
    audio_path: str | None,
    text_input: str,
    voice_label: str,
    speech_rate: int,
    resolution: str,
    render_mode: str,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("请上传人物视频。")
    if not audio_path and not (text_input or "").strip():
        raise gr.Error("请上传驱动音频，或输入需要朗读的文案。")

    video = Path(video_path).resolve()
    if not video.is_file():
        raise gr.Error("人物视频不存在，请重新上传。")

    running = subprocess.run(
        ["pgrep", "-f", "python -m scripts.inference"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if running.returncode == 0:
        raise gr.Error("已有一个生成任务正在运行。请等待它完成，避免 16GB 统一内存溢出。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video.stem).strip("._") or "digital_human"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = OUTPUT_DIR / f"{safe_stem}_{timestamp}.mp4"
    tts_temp: tempfile.TemporaryDirectory | None = None

    if audio_path:
        audio = Path(audio_path).resolve()
        if not audio.is_file():
            raise gr.Error("驱动音频不存在，请重新上传。")
    else:
        progress(0.01, desc="正在把文案转换成普通话语音……")
        tts_temp = tempfile.TemporaryDirectory(prefix=".tts-", dir=OUTPUT_DIR)
        try:
            audio = _synthesize_text(text_input, voice_label, speech_rate, Path(tts_temp.name))
        except subprocess.CalledProcessError as error:
            tts_temp.cleanup()
            details = (error.stderr or str(error)).strip()
            raise gr.Error(f"文本配音失败：{details}") from error

    env = os.environ.copy()
    env["LATENTSYNC_MAX_DIM"] = "1920" if resolution == "1080p" else "1280"
    env["PYTHONUNBUFFERED"] = "1"
    progress(0.05, desc="正在预处理视频并加载模型……")

    if render_mode.startswith("双机"):
        command = [
            str(ROOT_DIR / ".venv/bin/python"),
            str(ROOT_DIR / "dual_mac_render.py"),
            "--max-dim",
            env["LATENTSYNC_MAX_DIM"],
            str(video),
            str(audio),
            str(output),
        ]
    else:
        command = [str(RUN_SCRIPT), str(video), str(audio), str(output)]

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        state = {"phase": "startup", "batch": 0, "batches": 1}
        log_tail = []
        current_line = []
        assert process.stdout is not None
        while True:
            character = process.stdout.read(1)
            if character == "" and process.poll() is not None:
                if current_line:
                    line = "".join(current_line).strip()
                    _update_progress_from_log(line, state, progress)
                    log_tail.append(ANSI_ESCAPE.sub("", line))
                break
            if character in {"\r", "\n"}:
                if current_line:
                    line = "".join(current_line).strip()
                    _update_progress_from_log(line, state, progress)
                    log_tail.append(ANSI_ESCAPE.sub("", line))
                    log_tail = log_tail[-20:]
                    current_line.clear()
            else:
                current_line.append(character)
    finally:
        if tts_temp is not None:
            tts_temp.cleanup()

    if process.returncode != 0:
        details = "\n".join(log_tail)
        raise gr.Error(f"生成失败：\n{details}")

    progress(1.0, desc="生成完成")
    return str(output), str(output)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LatentSync Mac 数字人") as demo:
        gr.Markdown(
            """
            # LatentSync Mac 数字人
            上传一段正脸人物视频，再上传音频或输入文案，生成口型同步视频。

            **M4 16GB 推荐使用 720p。** 推理按 8 帧分批、20 步运行，首次生成及长视频需要较长时间。

            M4 的实测速度约为每 1 秒音频需要 12–15 分钟。建议先用 3–5 秒素材测试；页面一次只运行一个任务。
            """
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="人物视频", sources=["upload"])
                audio_input = gr.Audio(
                    label="驱动音频（与文案二选一，上传音频优先）",
                    type="filepath",
                    sources=["upload"],
                )
                text_input = gr.Textbox(
                    label="配音文案（未上传音频时使用）",
                    placeholder="请输入数字人需要说的话……",
                    lines=5,
                )
                with gr.Row():
                    voice = gr.Dropdown(
                        choices=list(TTS_VOICES),
                        value="婷婷（普通话女声）",
                        label="配音声音",
                    )
                    speech_rate = gr.Slider(
                        minimum=120,
                        maximum=260,
                        value=180,
                        step=10,
                        label="语速",
                    )
                preview_button = gr.Button("试听配音")
                voice_preview = gr.Audio(label="配音试听", type="filepath", autoplay=True)
                resolution = gr.Radio(
                    choices=["720p", "1080p"],
                    value="720p",
                    label="输出预处理规格",
                    info="1080p 更慢且占用更多统一内存。",
                )
                render_mode = gr.Radio(
                    choices=["双机加速（推荐）", "仅本机"],
                    value="双机加速（推荐）",
                    label="计算模式",
                    info="双机模式会自动拆分、分发并合并视频。",
                )
                generate_button = gr.Button("生成数字人视频", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="生成结果")
                download_output = gr.File(label="下载 MP4")

        gr.Examples(
            examples=[
                ["assets/demo1_video.mp4", "assets/demo1_audio.wav", "", "婷婷（普通话女声）", 180, "720p", "双机加速（推荐）"],
                ["assets/demo2_video.mp4", None, "欢迎使用双机数字人视频生成工具。", "婷婷（普通话女声）", 180, "720p", "双机加速（推荐）"],
            ],
            inputs=[video_input, audio_input, text_input, voice, speech_rate, resolution, render_mode],
        )

        preview_button.click(
            fn=preview_voice,
            inputs=[text_input, voice, speech_rate],
            outputs=[voice_preview],
        )

        generate_button.click(
            fn=generate_video,
            inputs=[video_input, audio_input, text_input, voice, speech_rate, resolution, render_mode],
            outputs=[video_output, download_output],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local LatentSync Mac interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=False,
        allowed_paths=[str(OUTPUT_DIR), str(ROOT_DIR / "assets")],
    )
