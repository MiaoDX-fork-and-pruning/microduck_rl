#!/usr/bin/env python3
"""Build a self-contained local HTML gallery from rollout videos."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi"}


def collect_manifest_videos(manifest_path: Path) -> tuple[list[tuple[str, Path]], dict[Path, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[tuple[str, Path]] = []
    evidence: dict[Path, dict[str, Any]] = {}
    for policy in manifest.get("policies", []):
        policy_id = policy.get("id")
        artifacts = policy.get("artifacts", {})
        raw_video = artifacts.get("diagnostic_video")
        if not isinstance(policy_id, str) or not policy_id or not isinstance(raw_video, str):
            raise ValueError("each manifest policy needs an id and diagnostic_video")
        video = Path(raw_video).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"{policy_id}: diagnostic video is missing or unsupported: {video}")
        evaluation: dict[str, Any] = {}
        raw_evaluation = artifacts.get("evaluation_report")
        if isinstance(raw_evaluation, str):
            evaluation_path = Path(raw_evaluation).expanduser().resolve()
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        entries.append((policy_id, video))
        evidence[video] = {
            "task": policy.get("task", ""),
            "accepted": policy.get("accepted"),
            "evaluation": evaluation,
            "hashes": policy.get("sha256", {}),
        }
    if not entries:
        raise ValueError("manifest.policies must contain at least one policy")
    return entries, evidence


def collect_videos(inputs: list[Path]) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in inputs:
        root = root.expanduser().resolve()
        if root.is_file() and root.suffix.lower() in VIDEO_SUFFIXES:
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(p for p in root.rglob("*")
                               if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        else:
            raise SystemExit(f"input does not exist or is not a video: {root}")
        for video in candidates:
            video = video.resolve()
            if video in seen:
                continue
            seen.add(video)
            if root.is_file():
                task = video.stem
            else:
                relative = video.relative_to(root)
                task = relative.parts[0] if len(relative.parts) > 1 else root.name
            entries.append((task, video))
    return entries


def safe_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in value)
    return cleaned.strip().replace(" ", "_") or "untitled"


def _evidence_html(data: dict[str, Any]) -> str:
    if not data:
        return ""
    evaluation = data.get("evaluation", {})
    accepted = "accepted" if data.get("accepted") is True else "not accepted"
    metrics = [f"status: {accepted}"]
    if isinstance(evaluation.get("success_rate"), (int, float)):
        metrics.append(f"success: {evaluation['success_rate']:.1%}")
    if isinstance(evaluation.get("main_task_metric"), (int, float)):
        metrics.append(f"main metric: {evaluation['main_task_metric']:.4g}")
    penalties = evaluation.get("penalty_terms", {})
    penalty_text = ", ".join(
        f"{name}={value:.4g}" for name, value in sorted(penalties.items())
        if isinstance(value, (int, float))
    )
    review = evaluation.get("video_review", "")
    failure = evaluation.get("failure_note", "none") or "none"
    hashes = data.get("hashes", {})
    hash_rows = "".join(
        f"<dt>{html.escape(str(name))}</dt><dd><code>{html.escape(str(value))}</code></dd>"
        for name, value in sorted(hashes.items())
    )
    task = html.escape(str(data.get("task", "")))
    return (
        f'<dl class="metrics"><dt>Task</dt><dd>{task}</dd>'
        f'<dt>Metrics</dt><dd>{html.escape(" | ".join(metrics))}</dd>'
        f'<dt>Penalties</dt><dd>{html.escape(penalty_text or "none")}</dd>'
        f'<dt>Video review</dt><dd>{html.escape(str(review))}</dd>'
        f'<dt>Failure note</dt><dd>{html.escape(str(failure))}</dd>'
        f'{hash_rows}</dl>'
    )


def build_gallery(
    entries: list[tuple[str, Path]],
    output: Path,
    title: str,
    evidence: dict[Path, dict[str, Any]] | None = None,
) -> Path:
    video_root = output / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    used_names: set[Path] = set()
    for index, (task, source) in enumerate(entries, start=1):
        task_dir = video_root / safe_name(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        destination = task_dir / source.name
        if destination in used_names or destination.exists():
            destination = task_dir / f"{index:03d}_{source.name}"
        shutil.copy2(source, destination)
        used_names.add(destination)
        rel = destination.relative_to(output).as_posix()
        label = html.escape(task)
        filename = html.escape(source.name)
        details = _evidence_html((evidence or {}).get(source.resolve(), {}))
        cards.append(
            f'<article class="card" data-task="{label.lower()}">'
            f'<h2>{label}</h2><video controls preload="metadata" src="{html.escape(rel)}"></video>'
            f'<p>{filename}</p>{details}</article>'
        )

    tasks = sorted({task for task, _ in entries}, key=str.casefold)
    options = "".join(f'<option value="{html.escape(task.lower())}">{html.escape(task)}</option>'
                      for task in tasks)
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; font: 16px/1.4 system-ui,sans-serif; }}
body {{ margin:0; padding:24px; background:#111827; color:#e5e7eb; }}
header {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:20px; }}
h1 {{ margin:0; font-size:1.35rem; }}
select,input {{ background:#1f2937; color:inherit; border:1px solid #4b5563; padding:8px; border-radius:5px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }}
.card {{ background:#1f2937; border:1px solid #374151; padding:12px; border-radius:6px; }}
.card h2 {{ margin:0 0 8px; font-size:1rem; }}
video {{ display:block; width:100%; aspect-ratio:4/3; background:#000; object-fit:contain; }}
.card p {{ color:#9ca3af; font-size:.8rem; overflow-wrap:anywhere; }}
.metrics {{ display:grid; grid-template-columns:max-content 1fr; gap:5px 10px; font-size:.78rem; }}
.metrics dt {{ color:#9ca3af; }} .metrics dd {{ margin:0; min-width:0; overflow-wrap:anywhere; }}
code {{ font-size:.72rem; }}
</style></head><body>
<header><h1>{html.escape(title)}</h1>
<label>Task <select id="task"><option value="">All</option>{options}</select></label>
<label>Search <input id="search" placeholder="filename or task"></label>
<span id="count"></span></header>
<main class="grid">{"".join(cards)}</main>
<script>
const cards=[...document.querySelectorAll('.card')], task=document.querySelector('#task'), search=document.querySelector('#search'), count=document.querySelector('#count');
function filter() {{ const t=task.value, q=search.value.toLowerCase(); let n=0; cards.forEach(c=>{{const ok=(!t||c.dataset.task===t)&&(!q||c.textContent.toLowerCase().includes(q)); c.hidden=!ok; if(ok)n++;}}); count.textContent=`${{n}} / ${{cards.length}} videos`; }}
task.onchange=search.oninput=filter; filter();
</script></body></html>'''
    index = output / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="video files or directories")
    parser.add_argument("--manifest", type=Path, help="validated specialist artifact manifest")
    parser.add_argument("-o", "--output", type=Path, default=Path("artifacts/video-gallery"))
    parser.add_argument("--title", default="Microduck Policy Video Review")
    args = parser.parse_args()
    entries = collect_videos(args.inputs) if args.inputs else []
    evidence: dict[Path, dict[str, Any]] = {}
    if args.manifest:
        try:
            manifest_entries, evidence = collect_manifest_videos(args.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        known = {source.resolve() for _, source in entries}
        entries.extend(entry for entry in manifest_entries if entry[1].resolve() not in known)
    if not entries:
        parser.error("provide video inputs or --manifest")
    index = build_gallery(entries, args.output.expanduser().resolve(), args.title, evidence)
    print(f"built {len(entries)} video(s) in {index.parent}")
    print(f"open: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
