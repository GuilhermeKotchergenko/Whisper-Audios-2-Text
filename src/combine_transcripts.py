#!/usr/bin/env python3
"""Build one LLM-ready transcript from every individual transcription result."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COMBINED_JSON_NAME = "_all_transcripts.json"
COMBINED_MARKDOWN_NAME = "_all_transcripts.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine all individual transcription JSON files into one LLM-ready transcript."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    return parser.parse_args()


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == COMBINED_JSON_NAME:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Skipping invalid JSON result: %s", path.name)
            continue
        if not isinstance(result, dict) or not result.get("transcript"):
            logging.warning("Skipping result without a transcript: %s", path.name)
            continue
        results.append(result)
    return sorted(results, key=lambda result: result.get("transcribed_at", ""))


def combined_transcript(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Audio: {result.get('source_file', 'unknown')} | Transcribed: {result.get('transcribed_at', 'unknown')}]\n"
        f"{result['transcript']}"
        for result in results
    )


def write_combined_transcript(results_dir: Path) -> tuple[Path, Path, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_dir)
    transcript = combined_transcript(results)
    generated_at = datetime.now(UTC).isoformat()

    json_path = results_dir / COMBINED_JSON_NAME
    markdown_path = results_dir / COMBINED_MARKDOWN_NAME
    payload = {
        "generated_at": generated_at,
        "audio_count": len(results),
        "transcript": transcript,
        "sources": [
            {
                "source_file": result.get("source_file"),
                "transcribed_at": result.get("transcribed_at"),
                "archived_path": result.get("archived_path"),
            }
            for result in results
        ],
        "llm_input": {
            "instruction": "Use the following concatenated WhatsApp voice-note transcripts as input. Preserve source labels, uncertainty, dates, people, and action items. Do not invent details.",
            "transcript": transcript,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sections = "\n\n".join(
        "\n".join(
            [
                f"## {result.get('source_file', 'Unknown audio')}",
                "",
                f"Transcribed: {result.get('transcribed_at', 'unknown')}",
                "",
                result["transcript"],
            ]
        )
        for result in results
    ) or "_No individual transcriptions are available yet._"
    markdown_path.write_text(
        "\n".join(
            [
                "# All WhatsApp audio transcripts",
                "",
                f"Generated: {generated_at}",
                f"Audio count: {len(results)}",
                "",
                sections,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path, len(results)


def main() -> int:
    args = parse_args()
    json_path, markdown_path, count = write_combined_transcript(args.results_dir)
    print(f"Combined {count} transcript(s):")
    print(f"  - {json_path}")
    print(f"  - {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
