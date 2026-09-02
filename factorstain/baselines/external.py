from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .base import AcquisitionMethod, FitContext, MethodUnavailable
from .classical import ScannerTransformBank, as_uint8_rgb


class OfficialSubprocessMethod(AcquisitionMethod):
    """Dependency-isolated adapter for an official neural image implementation.

    The official environment command is deliberately explicit. It receives a JSON
    request and must produce the declared output. This prevents legacy dependency
    pins from mutating FactorStain's environment and makes the exact command part of
    provenance. No fallback model is substituted when the command is unavailable.
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
        self.command = command or os.getenv(env_name, "")
        self.append_scanner_lut = append_scanner_lut
        self.scanner_bank = ScannerTransformBank("lut", grid_size)

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
            raise MethodUnavailable(
                f"{self.method_name} official subprocess failed ({completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )

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
        if self.append_scanner_lut:
            self.scanner_bank.fit(
                context.scanner_pairs, context.image_size, context.seed
            )

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
            self._run("infer", request_path)
            if not output_path.exists():
                raise MethodUnavailable(
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
