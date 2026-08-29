#!/usr/bin/env python3
"""Build a self-contained local HTML gallery from rollout videos."""

from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi"}


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


def build_gallery(entries: list[tuple[str, Path]], output: Path, title: str) -> Path:
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
        cards.append(
            f'<article class="card" data-task="{label.lower()}">'
            f'<h2>{label}</h2><video controls preload="metadata" src="{html.escape(rel)}"></video>'
            f'<p>{filename}</p></article>'
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
    parser.add_argument("inputs", nargs="+", type=Path, help="video files or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("artifacts/video-gallery"))
    parser.add_argument("--title", default="Microduck Policy Video Review")
    args = parser.parse_args()
    entries = collect_videos(args.inputs)
    if not entries:
        raise SystemExit("no videos found (supported: mp4, webm, mov, m4v, avi)")
    index = build_gallery(entries, args.output.expanduser().resolve(), args.title)
    print(f"built {len(entries)} video(s) in {index.parent}")
    print(f"open: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
