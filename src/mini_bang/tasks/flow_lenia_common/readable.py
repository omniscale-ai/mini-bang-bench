"""Readable rendering of a VIA export; the gzipped export stays the only source.

The rendering uses the annotation language; its labels come from the vocabulary file
next to the annotation sources (``annotations.via_vocabulary``).
"""
from __future__ import annotations

from pathlib import Path
import textwrap

from mini_bang.tasks.flow_lenia_common.annotations import raw_step, via_vocabulary


def readable_via(via, *, title: str) -> str:
    labels = via_vocabulary()["readable"]
    names = {fid: Path(item["fname"]).stem for fid, item in via["file"].items()}
    video = {vid: names[str(view["fid_list"][0])] for vid, view in via["view"].items()}
    grouped = {}
    for annotation_id, item in via["metadata"].items():
        grouped.setdefault(video[str(item["vid"])], []).append((annotation_id, item))
    lines = [title, labels["time_scale"], labels["generated_notice"], ""]
    for name in sorted(grouped):
        lines += [f"=== {name} ===", ""]
        for annotation_id, item in sorted(grouped[name], key=lambda pair: (pair[1].get("z") or [-1])[0]):
            times, attrs = item.get("z", []), item.get("av", {})
            if not times:
                lines += [labels["video_note"], *wrap(attrs.get("3", ""))]
            else:
                separator = labels["range_separator"]
                seconds = separator.join(f"{value:g}" for value in times)
                steps = separator.join(str(raw_step(value)) for value in times)
                kind = attrs.get("1") or labels["untyped"]
                point = labels["point_marker"] if len(times) == 1 else ""
                lines += [labels["interval_line"].format(kind=kind, point=point, seconds=seconds, steps=steps)]
                if attrs.get("2"):
                    lines += wrap(attrs["2"])
            lines += [labels["annotation_id"].format(annotation_id=annotation_id), ""]
    return "\n".join(lines).rstrip() + "\n"


def wrap(text):
    return [line for paragraph in str(text).strip().splitlines()
            for line in (textwrap.wrap(paragraph, 100) or [""])]
