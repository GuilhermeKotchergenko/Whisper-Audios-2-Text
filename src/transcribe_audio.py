#!/usr/bin/env python3
"""Local WhatsApp-audio transcription, with no LLM call.

Drop .ogg files in a folder and run this command once, or keep it running with
--watch. Results are JSON documents designed to be passed to another LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from combine_transcripts import write_combined_transcript

SUPPORTED_AUDIO_EXTENSIONS = {".ogg", ".oga", ".opus"}
PT_BR_PROMPT = (
    "Transcrição em português brasileiro. Preserve nomes próprios, siglas, "
    "marcas, termos técnicos e expressões coloquiais brasileiras."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe WhatsApp voice notes locally with Whisper."
    )
    parser.add_argument("input", type=Path, help="An audio file or a folder containing audio files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/archive"))
    parser.add_argument("--model", default="large-v3", help="Whisper model, e.g. medium or large-v3.")
    parser.add_argument("--language", default="pt", help="Spoken language. Defaults to Portuguese (pt).")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Inference device. auto prefers CUDA and falls back to CPU.")
    parser.add_argument("--beam-size", type=int, default=5, help="Decoding search width. Higher can improve accuracy but is slower.")
    parser.add_argument("--initial-prompt", default=PT_BR_PROMPT, help="Vocabulary guidance for the transcription model.")
    parser.add_argument("--watch", action="store_true", help="Keep watching an input folder for new files.")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--min-file-age", type=float, default=5.0, help="Wait for a file to be this old before processing it.")
    parser.add_argument("--keep-source", action="store_true", help="Do not move successfully processed audio to the archive.")
    parser.add_argument("--no-notify", action="store_true", help="Do not show desktop notifications.")
    return parser.parse_args()


def audio_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file: {source}. Expected .ogg, .oga, or .opus.")
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {source}")


def load_model(model_name: str, device_preference: str) -> tuple[Any, str, str]:
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency. Run: python3 -m pip install -r requirements.txt"
        ) from error

    cuda_available = ctranslate2.get_cuda_device_count() > 0
    if device_preference == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = device_preference
    if device == "cuda" and not cuda_available:
        if device_preference == "cuda":
            raise RuntimeError(
                "CUDA was requested but no CUDA device is available to CTranslate2. "
                "Check NVIDIA drivers and CUDA/cuDNN libraries, or use --device cpu."
            )
        device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except RuntimeError as error:
        if device_preference == "auto" and device == "cuda":
            logging.warning("CUDA could not load (%s). Falling back to CPU int8.", error)
            return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu", "int8"
        raise
    return model, device, compute_type


def transcribe(
    model: Any,
    audio_path: Path,
    language: str | None,
    beam_size: int,
    initial_prompt: str | None,
) -> dict[str, Any]:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
    )
    entries = [
        {
            "start_seconds": round(segment.start, 2),
            "end_seconds": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        for segment in segments
        if segment.text.strip()
    ]
    transcript = " ".join(segment["text"] for segment in entries)
    return {
        "source_file": audio_path.name,
        "source_path": str(audio_path.resolve()),
        "transcribed_at": datetime.now(UTC).isoformat(),
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 3),
        "transcript": transcript,
        "segments": entries,
        "llm_input": {
            "instruction": "Use this WhatsApp voice-note transcript as input. Preserve uncertainty and do not invent details.",
            "transcript": transcript,
        },
    }


def output_path(output_dir: Path, audio_path: Path) -> Path:
    return output_dir / f"{audio_path.stem}.json"


def markdown_path(output_dir: Path, audio_path: Path) -> Path:
    return output_dir / f"{audio_path.stem}.md"


def archive_audio(audio_path: Path, archive_dir: Path) -> Path:
    dated_folder = archive_dir / datetime.now().strftime("%Y-%m-%d")
    dated_folder.mkdir(parents=True, exist_ok=True)
    destination = dated_folder / audio_path.name
    counter = 2
    while destination.exists():
        destination = dated_folder / f"{audio_path.stem}-{counter}{audio_path.suffix}"
        counter += 1
    return Path(shutil.move(str(audio_path), str(destination)))


def write_markdown(result: dict[str, Any], destination: Path) -> None:
    archived_path = result.get("archived_path", "Not archived")
    segments = "\n".join(
        f"- `{segment['start_seconds']:.2f}s–{segment['end_seconds']:.2f}s` {segment['text']}"
        for segment in result["segments"]
    ) or "_No speech was detected._"
    destination.write_text(
        "\n".join(
            [
                f"# {result['source_file']}",
                "",
                f"- Transcribed: {result['transcribed_at']}",
                f"- Language: {result['detected_language']} ({result['language_probability']:.1%} confidence)",
                f"- Archived audio: `{archived_path}`",
                "",
                "## Transcript",
                "",
                result["transcript"] or "_No speech was detected._",
                "",
                "## Timestamped segments",
                "",
                segments,
                "",
            ]
        ),
        encoding="utf-8",
    )


def notify(audio_name: str, transcript: str) -> None:
    preview = transcript[:180] or "No speech was detected."
    try:
        subprocess.run(
            ["notify-send", "--app-name=Whisper Audios", "Audio transcribed", f"{audio_name}\n{preview}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logging.warning("Desktop notifications are unavailable; transcription was still saved.")


def process_files(
    model: Any,
    source: Path,
    output_dir: Path,
    archive_dir: Path,
    language: str | None,
    beam_size: int,
    initial_prompt: str | None,
    model_name: str,
    device: str,
    compute_type: str,
    min_file_age: float,
    archive_sources: bool,
    show_notifications: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for audio_path in audio_files(source):
        if time.time() - audio_path.stat().st_mtime < min_file_age:
            logging.info("Waiting for %s to finish copying", audio_path.name)
            continue
        destination = output_path(output_dir, audio_path)
        if destination.exists():
            logging.info("Skipping %s; %s already exists", audio_path.name, destination.name)
            continue
        logging.info("Transcribing %s", audio_path.name)
        result = transcribe(model, audio_path, language, beam_size, initial_prompt)
        result["transcription_engine"] = {
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "language_requested": language,
        }
        if archive_sources:
            archived_path = archive_audio(audio_path, archive_dir)
            result["archived_path"] = str(archived_path.resolve())
        else:
            result["archived_path"] = None
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(result, markdown_path(output_dir, audio_path))
        if show_notifications:
            notify(audio_path.name, result["transcript"])
        logging.info("Wrote %s", destination)
        count += 1
    if count:
        _, combined_markdown, combined_count = write_combined_transcript(output_dir)
        logging.info("Updated %s with %s transcript(s)", combined_markdown.name, combined_count)
    return count


def main() -> int:
    args = parse_args()
    if args.watch:
        args.input.mkdir(parents=True, exist_ok=True)
    elif not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    if args.poll_seconds <= 0 or args.min_file_age < 0 or args.beam_size < 1:
        raise ValueError("--poll-seconds and --beam-size must be greater than zero; --min-file-age cannot be negative.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    model, device, compute_type = load_model(args.model, args.device)
    logging.info("Using %s with %s (%s)", args.model, device, compute_type)
    while True:
        process_files(
            model,
            args.input,
            args.output_dir,
            args.archive_dir,
            args.language,
            args.beam_size,
            args.initial_prompt,
            args.model,
            device,
            compute_type,
            args.min_file_age if args.watch else 0,
            not args.keep_source,
            not args.no_notify,
        )
        if not args.watch:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
