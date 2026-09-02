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

Legacy neural baselines run through an isolated command configured as
`FACTORSTAIN_<METHOD>_COMMAND`. The command receives one of:

```text
<command> fit   --request /absolute/path/to/fit_request.json
<command> infer --request /absolute/path/to/request.json
```

The fit request points to the strict training manifest and reference policy. The infer
request points to one source PNG and declares the output PNG. Adapter commands must use
the official checkout, record their environment and upstream commit, and must never read
the evaluation manifest during fitting. A missing command or output fails closed and is
reported as `UNAVAILABLE`; FactorStain never substitutes another network.

Environment sketches live in `env_specs/`. Upstream requirements remain authoritative.
Phaet and Mascaret require individual acceptance of Waiv's gated non-commercial terms;
their exact Hugging Face revision must be recorded after authenticated download.
