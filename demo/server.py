#!/usr/bin/env python3
"""Zero-dependency local server for the animated DeCal demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decal.allocators.base import BaseAllocator
from decal.orchestrator import DeCalOrchestrator, OrchestratorConfig
from decal.system_bridge import KVModelSpec, tokens_to_gib


SCENARIOS = {
    "sudden": "Sudden shift",
    "gradual": "Gradual drift",
    "recovery": "Shift and recovery",
}
N_WINDOWS = 64
N_BINS = 1200


def _workload_shape(t: int, scenario: str, magnitude: float) -> tuple[float, float, float]:
    """Return mean, standard deviation, and heavy-tail weight for one window."""
    baseline = 230.0
    shift = 260.0 * magnitude

    if scenario == "gradual":
        progress = min(max((t - 16) / 28.0, 0.0), 1.0)
        mean = baseline + shift * progress
        tail = 0.04 + 0.10 * progress
    elif scenario == "recovery":
        active = 1.0 if 20 <= t < 43 else 0.0
        mean = baseline + shift * active
        tail = 0.15 if active else 0.04
    else:
        active = 1.0 if t >= 24 else 0.0
        mean = baseline + shift * active
        tail = 0.15 if active else 0.04

    sigma = 44.0 + 18.0 * tail
    return mean, sigma, tail


def _make_pdf(t: int, scenario: str, magnitude: float) -> list[float]:
    mean, sigma, tail_weight = _workload_shape(t, scenario, magnitude)
    x = np.arange(N_BINS, dtype=np.float64)
    body = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    tail_mean = min(mean * 1.48 + 30.0, N_BINS - 80.0)
    tail = np.exp(-0.5 * ((x - tail_mean) / (sigma * 1.35)) ** 2)
    pdf = (1.0 - tail_weight) * body + tail_weight * tail
    pdf /= pdf.sum()
    return pdf.tolist()


def _orchestrator(decoupled: bool) -> DeCalOrchestrator:
    config = OrchestratorConfig(
        transfer_top_k=0,
        detector_type="ewma",
        stat_type="p95",
        ph_alpha=0.32,
        ph_lambda=7.0,
        ph_delta=0.35,
        alloc_alpha=0.60,
        base_estimator="scalar_ewma",
        sf_boost=1.08,
        sf_severe_boost=1.18,
        sf_hold_windows=2,
        sf_decay_rate=0.35,
        calibration_size=6,
    )
    orchestrator = DeCalOrchestrator(config)
    orchestrator.allocator.decouple_calibration = decoupled
    # Use the paper's reset-and-seed recovery path in both arms; only the score
    # reference differs between the two controllers.
    orchestrator.allocator.reset_calibration_on_drift = True
    return orchestrator


def _downsample_pdf(pdf: list[float], n_bars: int = 30) -> list[float]:
    values = np.array([chunk.sum() for chunk in np.array_split(np.asarray(pdf), n_bars)])
    peak = float(values.max())
    return (values / peak).round(4).tolist() if peak > 0 else values.tolist()


def simulate(scenario: str, magnitude: float) -> dict:
    if scenario not in SCENARIOS:
        scenario = "sudden"
    magnitude = min(max(magnitude, 0.6), 1.6)

    decal = _orchestrator(decoupled=True)
    coupled = _orchestrator(decoupled=False)
    model_spec = KVModelSpec()
    windows = []
    last_drift_count = 0

    for t in range(N_WINDOWS):
        pdf = _make_pdf(t, scenario, magnitude)
        actual_p99 = BaseAllocator.pdf_percentile(pdf, 0.99)
        result = decal.process_window(0, t, pdf)
        coupled_result = coupled.process_window(0, t, pdf)

        drift_count = int(result.metadata.get("drift_count", 0))
        drift = drift_count > last_drift_count
        last_drift_count = drift_count
        budget = float(result.budget)
        coupled_budget = float(coupled_result.budget)
        tail_index = min(max(int(math.floor(budget)) + 1, 0), len(pdf))
        request_oom = float(sum(pdf[tail_index:]))
        waste = max(budget - actual_p99, 0.0) / budget if budget > 0 else 0.0
        coupled_waste = (
            max(coupled_budget - actual_p99, 0.0) / coupled_budget
            if coupled_budget > 0
            else 0.0
        )

        windows.append(
            {
                "t": t,
                "actual": round(actual_p99, 2),
                "budget": round(budget, 2),
                "coupledBudget": round(coupled_budget, 2),
                "base": round(float(result.metadata.get("base_budget", 0.0)), 2),
                "protection": round(
                    float(result.metadata.get("base_budget", 0.0))
                    * max(float(result.safety_factor) - 1.0, 0.0),
                    2,
                ),
                "correction": round(float(result.metadata.get("correction", 0.0)), 2),
                "coupledCorrection": round(
                    float(coupled_result.metadata.get("correction", 0.0)), 2
                ),
                "safetyFactor": round(float(result.safety_factor), 3),
                "score": round(
                    actual_p99 - float(result.metadata.get("base_budget", 0.0)), 2
                ),
                "coupledScore": round(actual_p99 - coupled_budget, 2),
                "drift": drift,
                "oom": actual_p99 > budget,
                "coupledOom": actual_p99 > coupled_budget,
                "requestOom": round(request_oom * 100.0, 3),
                "waste": round(waste * 100.0, 2),
                "coupledWaste": round(coupled_waste * 100.0, 2),
                "kvGiB": round(tokens_to_gib(budget, model_spec), 3),
                "actualGiB": round(tokens_to_gib(actual_p99, model_spec), 3),
                "histogram": _downsample_pdf(pdf),
            }
        )

    for index, row in enumerate(windows):
        next_row = windows[min(index + 1, len(windows) - 1)]
        row["nextCorrection"] = next_row["correction"]
        row["coupledNextCorrection"] = next_row["coupledCorrection"]

    evaluated = windows[1:]
    summary = {
        "coverage": round(100.0 * sum(not row["oom"] for row in evaluated) / len(evaluated), 1),
        "coupledCoverage": round(
            100.0 * sum(not row["coupledOom"] for row in evaluated) / len(evaluated), 1
        ),
        "averageWaste": round(sum(row["waste"] for row in evaluated) / len(evaluated), 1),
        "coupledAverageWaste": round(
            sum(row["coupledWaste"] for row in evaluated) / len(evaluated), 1
        ),
        "drifts": sum(row["drift"] for row in windows),
    }
    return {
        "scenario": scenario,
        "scenarioLabel": SCENARIOS[scenario],
        "magnitude": magnitude,
        "windows": windows,
        "summary": summary,
        "model": model_spec.to_dict(),
    }


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/simulate":
            params = parse_qs(parsed.query)
            scenario = params.get("scenario", ["sudden"])[0]
            try:
                magnitude = float(params.get("magnitude", ["1.0"])[0])
            except ValueError:
                magnitude = 1.0
            self._send_json(simulate(scenario, magnitude))
            return

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/standalone.html": ("standalone.html", "text/html; charset=utf-8"),
            "/preview.gif": ("preview.gif", "image/gif"),
        }
        if parsed.path in static_files:
            filename, content_type = static_files[parsed.path]
            content = (Path(__file__).parent / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404)

    def _send_json(self, payload: dict) -> None:
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        if os.environ.get("DECAL_DEMO_VERBOSE"):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the animated DeCal demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"DeCal demo: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
