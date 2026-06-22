#!/usr/bin/env python3
"""Build a self-contained replay page with embedded default traces."""

from __future__ import annotations

import json
from pathlib import Path

from server import simulate


DEMO_DIR = Path(__file__).resolve().parent
FIELDS = (
    "t",
    "actual",
    "budget",
    "base",
    "safetyFactor",
    "score",
    "nextCorrection",
    "drift",
    "oom",
    "waste",
)


def main() -> None:
    payload = {}
    for scenario in ("sudden", "gradual", "recovery"):
        replay = simulate(scenario, 1.0)
        payload[scenario] = {
            "summary": replay["summary"],
            "windows": [
                [window[field] for field in FIELDS] for window in replay["windows"]
            ],
        }

    source = (DEMO_DIR / "index.html").read_text(encoding="utf-8")
    embedded = (
        "<script>window.embeddedReplayData="
        + json.dumps(payload, separators=(",", ":"))
        + ";</script>"
    )
    source = source.replace("<!-- EMBEDDED_REPLAY_DATA -->", embedded, 1)
    (DEMO_DIR / "standalone.html").write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
