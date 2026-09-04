from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
from PIL import Image

from .base import AcquisitionMethod, FitContext, MethodUnavailable
from .classical import ScannerTransformBank, as_uint8_rgb


class OfficialAdapterExecutionError(RuntimeError):
    """An installed official adapter started but failed to execute."""


class OfficialSubprocessMethod(AcquisitionMethod):
    """Subprocess adapter for a pinned official neural image implementation.

    The command receives a JSON request and must produce the declared output. The
    bundled command uses FactorStain's environment; a custom isolated environment may
    be supplied explicitly. No fallback model is substituted on failure.
    """

    def __init__(
        self,
        method_name: str,
        command: str | None = None,
        append_scanner_lut: bool = True,
        grid_size: int = 17,
    ) -> None:
        self.method_name = method_name
        env_name = f"FACTORSTAIN_{method_name.upper()}_COMMAND"
        configured = command or os.getenv(env_name, "")
        bundled = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_official_baseline_adapter.py"
        )
        self.command = configured or shlex.join(
            [sys.executable, str(bundled), "--method", method_name]
        )
        self.use_persistent_server = (
            not configured or os.getenv("FACTORSTAIN_ADAPTER_PERSISTENT", "0") == "1"
        )
        self.append_scanner_lut = append_scanner_lut
        self.scanner_bank = ScannerTransformBank("lut", grid_size)
        self._server: subprocess.Popen[str] | None = None
        self._server_log: IO[str] | None = None

    def _run(self, mode: str, request: Path) -> None:
        if not self.command:
            raise MethodUnavailable(
                f"{self.method_name}: isolated official adapter command is unset; "
                f"set FACTORSTAIN_{self.method_name.upper()}_COMMAND after running "
                "third_party/fetch_baselines.py and creating its environment"
            )
        completed = subprocess.run(
            [*shlex.split(self.command), mode, "--request", str(request)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise OfficialAdapterExecutionError(
                f"{self.method_name} official subprocess failed ({completed.returncode}): "
                f"{(completed.stderr or completed.stdout)[-4000:]}"
            )

    def _start_server(self) -> None:
        if self._server is not None:
            return
        request_path = self.checkpoint_dir / "serve_request.json"
        request_path.write_text(
            json.dumps(
                {
                    "method": self.method_name,
                    "checkpoint_dir": str(self.checkpoint_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log_path = self.checkpoint_dir / "inference_server.log"
        self._server_log = log_path.open("a", encoding="utf-8")
        self._server = subprocess.Popen(
            [
                *shlex.split(self.command),
                "serve",
                "--request",
                str(request_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._server_log,
            text=True,
            bufsize=1,
        )
        assert self._server.stdout is not None
        ready = self._server.stdout.readline()
        try:
            response = json.loads(ready)
        except json.JSONDecodeError as exc:
            self.close()
            raise OfficialAdapterExecutionError(
                f"{self.method_name} inference server did not start; see {log_path}"
            ) from exc
        if response.get("status") != "READY":
            self.close()
            raise OfficialAdapterExecutionError(
                f"{self.method_name} inference server failed: {response}"
            )

    def _infer(self, request: dict, request_path: Path) -> None:
        if not self.use_persistent_server:
            self._run("infer", request_path)
            return
        self._start_server()
        assert self._server is not None
        assert self._server.stdin is not None and self._server.stdout is not None
        self._server.stdin.write(json.dumps(request) + "\n")
        self._server.stdin.flush()
        line = self._server.stdout.readline()
        if not line:
            return_code = self._server.poll()
            self.close()
            raise OfficialAdapterExecutionError(
                f"{self.method_name} inference server exited unexpectedly "
                f"with status {return_code}"
            )
        response = json.loads(line)
        if response.get("status") != "COMPLETE":
            log_path = self.checkpoint_dir / "inference_server.log"
            raise OfficialAdapterExecutionError(
                f"{self.method_name} inference failed: "
                f"{response.get('error_type')}: {response.get('error')}; "
                f"full traceback: {log_path}"
            )

    def close(self) -> None:
        server = getattr(self, "_server", None)
        self._server = None
        if server is not None:
            try:
                if server.stdin is not None:
                    server.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
        server_log = getattr(self, "_server_log", None)
        if server_log is not None:
            server_log.close()
            self._server_log = None

    def _scanner_cache_path(self, context: FitContext) -> Path:
        columns = [
            "source_scanner",
            "target_scanner",
            "source_path",
            "target_path",
        ]
        pairs = context.scanner_pairs[columns].astype(str).sort_values(columns)
        digest = hashlib.sha256()
        digest.update(b"factorstain-scanner-lut-v1\n")
        digest.update(pairs.to_csv(index=False, lineterminator="\n").encode())
        digest.update(
            f"\n{context.image_size}:{context.seed}:{self.scanner_bank.grid_size}".encode()
        )
        return (
            context.output_dir
            / "shared_scanner_lut"
            / f"scanner_lut_{digest.hexdigest()[:16]}.npz"
        )

    def _fit_or_load_scanner_bank(self, context: FitContext) -> None:
        cache_path = self._scanner_cache_path(context)
        manifest_path = cache_path.with_suffix(".json")
        if cache_path.is_file() and manifest_path.is_file():
            self.scanner_bank.load(cache_path)
            return
        self.scanner_bank.fit(context.scanner_pairs, context.image_size, context.seed)
        temporary = cache_path.with_name(
            f".{cache_path.stem}.{os.getpid()}.tmp{cache_path.suffix}"
        )
        self.scanner_bank.save(temporary)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, cache_path)
        os.replace(temporary.with_suffix(".json"), manifest_path)

    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        self.context = context
        self.checkpoint_dir = context.output_dir / self.method_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        benchmark_output = Path(context.metadata["benchmark_output"])
        request = {
            "method": self.method_name,
            "seed": context.seed,
            "fast_dev_run": context.fast_dev_run,
            "image_size": context.image_size,
            "train_manifest": str(
                benchmark_output / "metadata" / "training_manifest.parquet"
            ),
            "validation_manifest": str(
                benchmark_output / "metadata" / "validation_manifest.parquet"
            ),
            "reference_policy": str(
                benchmark_output / "metadata" / "reference_policy.json"
            ),
            "scanner_fit_pairs": str(
                benchmark_output / "metadata" / "scanner_fit_pairs.parquet"
            ),
            "native_task": "paired_scanner_transfer"
            if self.method_name == "pix2pix"
            else "train_only_stain_translation",
            "checkpoint_dir": str(self.checkpoint_dir),
            "forbidden_target_ids_sha256": context.reference_policy[
                "forbidden_target_ids_sha256"
            ],
        }
        request_path = self.checkpoint_dir / "fit_request.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        self._run("fit", request_path)
        state_path = self.checkpoint_dir / "adapter_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.fit_fingerprint = str(state.get("fit_fingerprint", ""))
        else:
            self.fit_fingerprint = ""
        if not self.fit_fingerprint:
            self.fit_fingerprint = hashlib.sha256(
                request_path.read_bytes() + self.command.encode()
            ).hexdigest()
        if self.append_scanner_lut:
            self._fit_or_load_scanner_bank(context)

    def translate(
        self,
        source_image: Image.Image | np.ndarray,
        source_stain: str,
        source_scanner: str,
        target_stain: str,
        target_scanner: str,
        *,
        track: str = "C",
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory(
            prefix=f"factorstain-{self.method_name}-"
        ) as temp:
            temporary = Path(temp)
            source_path = temporary / "source.png"
            output_path = temporary / "output.png"
            Image.fromarray(as_uint8_rgb(source_image)).save(source_path)
            request = {
                "method": self.method_name,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "source_stain": str(source_stain),
                "target_stain": str(target_stain),
                "source_scanner": str(source_scanner),
                "target_scanner": str(target_scanner),
                "checkpoint_dir": str(self.checkpoint_dir),
                "track": track,
            }
            request_path = temporary / "request.json"
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            self._infer(request, request_path)
            if not output_path.exists():
                raise OfficialAdapterExecutionError(
                    f"{self.method_name} adapter did not create {output_path}"
                )
            generated = np.asarray(Image.open(output_path).convert("RGB")).copy()
        if track == "C" and self.append_scanner_lut:
            # The stain translator is trained across scanner-pooled stain domains. Its
            # output reference scanner is declared by the target-stain prototype.
            reference_scanner = str(
                self.context.reference_policy["stain_prototypes"][str(target_stain)][
                    "scanner_id"
                ]
            )
            generated = self.scanner_bank.apply(
                generated, reference_scanner, str(target_scanner)
            )
        return generated

    def supports_strict_composition(self) -> bool:
        return self.append_scanner_lut

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
