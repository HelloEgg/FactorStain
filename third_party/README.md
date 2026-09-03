# External baseline isolation

`fetch_baselines.py` clones only reviewed official repositories and checks out the
immutable commits in `factorstain.baselines.registry`. It does not install packages or
modify the FactorStain environment.

If a repository was copied as a vendored directory without its nested `.git` folder,
the fetcher compares every upstream-tracked file with a temporary checkout of the pinned
commit. A matching snapshot receives `.factorstain-source.json` and is reused; a modified
or incomplete snapshot is never overwritten and fails with recovery instructions.

```bash
python third_party/fetch_baselines.py --tier 1
```

Bundled neural adapters run through the same subprocess boundary as custom isolated
environments. The default command is `scripts/run_official_baseline_adapter.py`; it may
be overridden with `FACTORSTAIN_<METHOD>_COMMAND`. A command receives one of:

```text
<command> fit   --request /absolute/path/to/fit_request.json
<command> infer --request /absolute/path/to/request.json
<command> serve --request /absolute/path/to/serve_request.json
```

The fit request points to the strict training manifest and reference policy. The infer
request points to one source PNG and declares the output PNG. `serve` uses JSON-lines on
stdin/stdout so a target model remains resident during evaluation. Adapter commands use
the pinned official checkout, record their environment and upstream commit, and never
read the evaluation manifest during fitting. A missing command or output fails closed;
FactorStain never substitutes another network.

Run the source/runtime preflight with:

```bash
bash shell/setup_external_baselines.sh
```

This refreshes the editable FactorStain environment so `torchvision` and the other
declared runtime dependencies are present. Set `SKIP_PYTHON_SETUP=1` only when an
existing `.venv` has already been provisioned externally.

Training policy is frozen in `configs/external_neural_baselines.yaml`. Every stain target
uses the train-only prototype's scanner as its declared output scanner: StainNet uses
aligned-group pairs; StainGAN and CycleGAN use unpaired target banks; Pix2Pix uses
registered same-group/same-stain scanner pairs; HistAuGAN uses one multidomain model;
CAGAN and SAStainDiff use target banks. Stain outputs are then composed with the
training-only ScannerLUT for strict Track C.

The source repositories are registered as submodules. On a fresh checkout either run:

```bash
git submodule update --init --recursive
```

or use `fetch_baselines.py`. The fetcher also initializes pre-existing empty gitlink
directories, so manually deleting them is not required.

Environment sketches live in `env_specs/`. Upstream requirements remain authoritative.
Phaet and Mascaret require individual acceptance of Waiv's gated non-commercial terms;
their exact Hugging Face revision must be recorded after authenticated download.
