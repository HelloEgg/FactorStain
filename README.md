# FactorStain

FactorStain is a milestone-driven research implementation for factorizing pathology
appearance into ordered H&E staining and scanner-rendering operators:

\[
x_{s,q}=Q_q(S_s(m)).
\]

The decisive experiment is generation of stain/scanner pairs that are never jointly
observed during training. All milestone decisions are computed from measured outputs;
development runs are always marked invalid for scientific decision-making.

## Quick start

```bash
bash shell/setup.sh
bash shell/run_all.sh
```

Run a complete, small pipeline check with:

```bash
FAST_DEV_RUN=1 bash shell/run_all.sh
```

Individual stages are `shell/m0_probe.sh` through `shell/m5_external.sh`. Dataset and
cache locations are configured in `configs/paths.yaml`; every other setting can be
overridden with environment variables documented in the YAML files.

Before M0, an optional frozen-feature preflight inspects raw PLISM/MIDOG21 domains
and official Meta DINOv3 embeddings without training any FactorStain component:

```bash
bash shell/m_minus1_domain_audit.sh
FAST_DEV_RUN=1 bash shell/m_minus1_domain_audit.sh
```

Its one-page result is `outputs/m_minus1_domain_audit/figures/SUMMARY_DASHBOARD.png`.
Set `RESAMPLE=1` to rebuild deterministic sample manifests, `FORCE_REEXTRACT=1` to
replace matching feature caches, or `DINOV3_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m`
to select the larger official backbone. Gated model access uses `HF_TOKEN`.

## Scientific stages

| Stage | Question |
|---|---|
| M0 | Do frozen pathology foundation models encode stain/scanner information? |
| M1 | Does explicit factorization improve held-out stain×scanner generation? |
| M2 | Does physical S→Q ordering outperform reverse/parallel alternatives? |
| M3 | Do counterfactual sensitivities reproduce real factorial sensitivities? |
| M4 | Can FactorAdapter remove acquisition leakage while preserving biology? |
| M5 | Does the adapter improve leave-center/scanner-out robustness? |

Each stage writes `metrics.json`, `metrics.csv`, `GO_NOGO.json`, `REPORT.md`, a
resolved config, logs, figures, and checkpoints under `outputs/`. The current project
state is summarized in `outputs/MASTER_DASHBOARD.png` and `outputs/MASTER_REPORT.md`.

## Notes

- Neural training requires CUDA; expensive jobs never silently fall back to CPU.
- UNI and Virchow2 are loaded by their actual registries. Gated access failures are
  recorded as `BLOCKED_MODEL_ACCESS` and never substituted with an unrelated model.
- WSI pipelines cache coordinates and embeddings rather than millions of image tiles.
- Macenko normalization and OD-domain stain augmentation are implemented in
  `factorstain/data/stain.py`; the evaluation baseline registry accepts later
  HistoFS/FEATMAP-style corrections without changing evaluator code.
- Use `torchrun --standalone --nproc_per_node=4 ...` through the supplied shell scripts.
