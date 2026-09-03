#!/usr/bin/env python
"""Run a pinned external neural baseline through FactorStain's adapter contract."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factorstain.baselines.official_neural import (
    SUPPORTED_METHODS,
    AdapterRuntime,
    describe_plan,
    fit_adapter,
    infer_one,
)


def _request(path: str) -> dict:
    request_path = Path(path).resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"Adapter request does not exist: {request_path}")
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["_request_path"] = str(request_path)
    return payload


def _serve(method: str, request: dict) -> None:
    runtime = AdapterRuntime(method, request)
    print(json.dumps({"status": "READY", "method": method}), flush=True)
    for line in sys.stdin:
        try:
            item = json.loads(line)
            # Several pinned upstream factories print initialization messages. Keep
            # stdout machine-readable for the JSON-lines transport.
            with contextlib.redirect_stdout(sys.stderr):
                infer_one(method, item, runtime=runtime)
            response = {"status": "COMPLETE", "output_path": item["output_path"]}
        except Exception as exc:  # noqa: BLE001 - transport must report remote errors
            response = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(json.dumps(response), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("mode", choices=("fit", "infer", "serve", "plan"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = _request(args.request)
    if args.mode == "fit":
        fit_adapter(args.method, request)
    elif args.mode == "infer":
        infer_one(args.method, request)
    elif args.mode == "serve":
        _serve(args.method, request)
    else:
        print(json.dumps(describe_plan(args.method, request), indent=2))


if __name__ == "__main__":
    main()
