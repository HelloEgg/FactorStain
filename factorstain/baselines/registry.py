from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineSpec:
    method_name: str
    display_name: str
    citation: str
    paper_year: int | None
    official_repository: str
    source_commit: str
    implementation_source: str
    native_task: str
    supports_stain: bool
    supports_scanner: bool
    supports_unseen_composition: bool
    requires_target_domain_samples: bool
    primary_track: str
    tier: int
    method_type: str
    external_pretraining: str
    availability: str
    license: str
    strict_track_c_form: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _spec(
    method_name: str,
    display_name: str,
    citation: str,
    year: int | None,
    repo: str = "",
    commit: str = "",
    source: str = "REIMPLEMENTED",
    native: str = "stain normalization",
    stain: bool = True,
    scanner: bool = False,
    composition: bool = False,
    target: bool = True,
    track: str = "A",
    tier: int = 1,
    kind: str = "CLASSICAL",
    pretraining: str = "NONE",
    availability: str = "AVAILABLE",
    license_name: str = "See upstream",
    notes: str = "",
) -> BaselineSpec:
    adapted_stain_methods = {
        "histogram",
        "reinhard",
        "macenko",
        "vahadane",
        "stainnet",
        "staingan",
        "cyclegan",
        "histaugan",
        "cagan",
        "sastaindiff",
    }
    strict_form = (
        f"{display_name} + ScannerLUT (COMPOSITIONAL_ADAPTATION)"
        if method_name in adapted_stain_methods
        else display_name
        if composition or method_name == "noadapt"
        else "NOT_APPLICABLE"
    )
    return BaselineSpec(
        method_name,
        display_name,
        citation,
        year,
        repo,
        commit,
        source,
        native,
        stain,
        scanner,
        composition,
        target,
        track,
        tier,
        kind,
        pretraining,
        availability,
        license_name,
        strict_form,
        notes,
    )


BASELINES = {
    item.method_name: item
    for item in [
        _spec(
            "noadapt",
            "NoAdapt",
            "Identity/no intervention",
            None,
            native="identity",
            stain=False,
            target=False,
            track="A,B,C,D",
            notes="Negative control.",
        ),
        _spec(
            "histogram",
            "Histogram Matching",
            "scikit-image exposure.match_histograms",
            None,
            repo="https://scikit-image.org/docs/stable/api/skimage.exposure.html",
            source="OFFICIAL",
            composition=True,
            notes="Per-channel RGB quantile matching; Track C appends a training-only ScannerLUT.",
        ),
        _spec(
            "reinhard",
            "Reinhard",
            "Reinhard et al., Color Transfer between Images",
            2001,
            composition=True,
            notes="Lab mean/std transfer with epsilon-clamped standard deviations.",
        ),
        _spec(
            "macenko",
            "Macenko",
            "Macenko et al., A Method for Normalizing Histology Slides",
            2009,
            composition=True,
            notes="Optical-density SVD, alpha=1, beta=0.15; Track C appends ScannerLUT.",
        ),
        _spec(
            "vahadane",
            "Vahadane",
            "Vahadane et al., Structure-Preserving Color Normalization",
            2016,
            composition=True,
            notes="Sparse non-negative stain factorization via deterministic NMF; Track C appends ScannerLUT.",
        ),
        _spec(
            "rgb_affine",
            "RGB Affine",
            "Least-squares RGB color calibration",
            None,
            native="scanner harmonization",
            stain=False,
            scanner=True,
            composition=False,
            target=False,
            track="B",
            notes="Fitted from clean same-group/same-stain training pairs.",
        ),
        _spec(
            "polynomial",
            "Polynomial Color",
            "Polynomial color calibration",
            None,
            native="scanner harmonization",
            stain=False,
            scanner=True,
            composition=False,
            target=False,
            track="B",
            notes="Ten-term quadratic RGB design fitted on clean training pairs.",
        ),
        _spec(
            "scanner_lut",
            "3D LUT",
            "Trilinear 3D color lookup calibration",
            None,
            native="scanner harmonization",
            stain=False,
            scanner=True,
            composition=False,
            target=False,
            track="B",
            notes="Primary scanner module; smoothed residual grid fitted from clean training pairs.",
        ),
        _spec(
            "stainnet",
            "StainNet",
            "Kang et al., A Fast and Robust Stain Normalization Network",
            2021,
            "https://github.com/khtao/StainNet",
            "94c20b31c0784d0d49468265afdde3d131d6afc8",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            notes="Official model wrapped in an isolated subprocess; PLISM weights must be trained from allowed stain domains. Strict row is StainNet + ScannerLUT.",
        ),
        _spec(
            "staingan",
            "StainGAN",
            "Shaban et al., StainGAN: Stain Style Transfer for Digital Histological Images",
            2019,
            "https://github.com/xtarx/StainGAN",
            "743f5e8243a87485f4e4f5a2a4b9af75aa8b0414",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            notes="Unpaired stain translation; strict row appends ScannerLUT.",
        ),
        _spec(
            "cyclegan",
            "CycleGAN",
            "Zhu et al., Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
            2017,
            "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix",
            "2a7afba2895d52556dd5dfe07e8555ef657ced6f",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            license_name="BSD-3-Clause",
            notes="Official ResNet generator; train-only stain domains. Strict row appends ScannerLUT.",
        ),
        _spec(
            "pix2pix",
            "Pix2Pix",
            "Isola et al., Image-to-Image Translation with Conditional Adversarial Networks",
            2017,
            "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix",
            "2a7afba2895d52556dd5dfe07e8555ef657ced6f",
            "ADAPTED_OFFICIAL",
            native="paired scanner transfer",
            stain=False,
            scanner=True,
            composition=False,
            target=False,
            track="B",
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            license_name="BSD-3-Clause",
            notes="Only same-group/same-stain scanner pairs are valid; cross-stain is UNSUPPORTED_INVALID_ALIGNMENT.",
        ),
        _spec(
            "histaugan",
            "HistAuGAN",
            "Wagner et al., Structure-Preserving Multi-Domain Stain Color Augmentation",
            2021,
            "https://github.com/sophiajw/HistAuGAN",
            "fed016328ed941069e6fc0aef32f7f88bd624af4",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            notes="Train-only target domains; strict row appends ScannerLUT.",
        ),
        _spec(
            "cagan",
            "CAGAN",
            "Cong et al., Colour Adaptive Generative Networks for Stain Normalisation",
            2022,
            "https://github.com/thomascong121/CAGAN_Stain_Norm",
            "ca7bc786da1dd37034992d75ef96d3d4ab024f02",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=1,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            notes="Official model/data boundary adapted to PLISM; strict row appends ScannerLUT.",
        ),
        _spec(
            "sastaindiff",
            "SAStainDiff",
            "Yu et al., Self-supervised Stain Normalization by Stain Augmentation using DDPMs",
            2025,
            "https://github.com/yhuaishui/SAStainDiff",
            "0a33206998a129b0778c93467b4da8a491a7df50",
            "ADAPTED_OFFICIAL",
            composition=True,
            tier=2,
            kind="LEARNED PRIOR WORK",
            availability="REQUIRES_THIRD_PARTY_SETUP",
            notes="FULL retraining is expensive; pretrained external-data runs are separated. Strict row appends ScannerLUT.",
        ),
        _spec(
            "histofs",
            "HistoFS",
            "Raswa et al., Non-IID Histopathologic WSI Classification via Federated Style Transfer",
            2025,
            "https://github.com/lalakitchen/HistoFS",
            "ec944a25f073ffba4f6168cd44e76fd4c35b6c9d",
            "OFFICIAL",
            native="federated WSI feature augmentation",
            stain=False,
            scanner=False,
            composition=False,
            target=False,
            track="D",
            tier=2,
            kind="FEATURE/DOWNSTREAM",
            availability="NOT_APPLICABLE_TO_STRICT_IMAGE_COMPOSITION",
            license_name="MIT",
            notes="Feature-level federated MIL method, not an image counterfactual generator.",
        ),
        _spec(
            "scangen",
            "ScanGen",
            "Carloni et al., Pathology Foundation Models are Scanner Sensitive",
            2025,
            repo="https://arxiv.org/abs/2507.22092",
            source="REIMPLEMENTED",
            native="contrastive downstream feature projection",
            stain=False,
            scanner=True,
            target=False,
            track="D",
            tier=2,
            kind="FEATURE/DOWNSTREAM",
            notes="Paper-faithful 3-layer MLP and published attraction/repulsion loss; PLISM tissue CE replaces the paper's unavailable EGFR task to preserve biology; no official code found.",
        ),
        _spec(
            "featmap",
            "FEATMAP",
            "Donle et al., Targeted Correction of Acquisition Signatures Harmonizes Medical FM Embeddings",
            2026,
            repo="https://doi.org/10.64898/2026.07.02.736184",
            source="REIMPLEMENTED",
            native="paired affine embedding harmonization",
            stain=False,
            scanner=True,
            target=False,
            track="D",
            tier=2,
            kind="FEATURE/DOWNSTREAM",
            notes="Global affine least-squares mapping learned from same-morphology scanner pairs; no official code found.",
        ),
        _spec(
            "phaet",
            "Phaet",
            "Filiot et al., Robustifying Pathology Foundation Models via Fine-tuning",
            2026,
            repo="https://huggingface.co/wearewaiv/phaet",
            source="OFFICIAL",
            native="robust frozen pathology feature extractor",
            stain=False,
            scanner=False,
            target=False,
            track="D",
            tier=2,
            kind="FEATURE/DOWNSTREAM",
            pretraining="PRETRAINED_EXTERNAL_DATA",
            availability="GATED_MODEL_LICENSE_REQUIRED",
            license_name="Waiv non-commercial",
            notes="Official gated Phikon-v2 derivative; revision is resolved only after authenticated access.",
        ),
        _spec(
            "mascaret",
            "Mascaret",
            "Filiot et al., Robustifying Pathology Foundation Models via Fine-tuning",
            2026,
            "https://huggingface.co/wearewaiv/mascaret",
            source="OFFICIAL",
            native="robust frozen pathology feature extractor",
            stain=False,
            scanner=False,
            target=False,
            track="D",
            tier=2,
            kind="FEATURE/DOWNSTREAM",
            pretraining="PRETRAINED_EXTERNAL_DATA",
            availability="GATED_MODEL_LICENSE_REQUIRED",
            license_name="Waiv non-commercial",
            notes="Official gated Midnight-12k derivative; revision is resolved only after authenticated access.",
        ),
        _spec(
            "igan",
            "I-GAN",
            "Chen et al., Improving stain normalization ... identity loss model",
            2026,
            repo="https://doi.org/10.1177/20552076261438012",
            source="REIMPLEMENTED",
            composition=False,
            tier=3,
            kind="LEARNED PRIOR WORK",
            availability="NOT_REPRODUCIBLE_FROM_AVAILABLE_RESOURCES",
            notes="Paper exists, but no official code or sufficiently complete public implementation was found; no architecture is invented.",
        ),
        _spec(
            "joint",
            "Joint",
            "FactorStain M1 internal joint baseline",
            2026,
            source="OFFICIAL",
            native="joint stain/scanner renderer",
            scanner=True,
            composition=True,
            target=False,
            track="A,B,C",
            tier=1,
            kind="OURS / INTERNAL",
            notes="Existing M1 checkpoint and exact optimization budget.",
        ),
        _spec(
            "parallel",
            "OURS-Parallel",
            "FactorStain M1 parallel factor baseline",
            2026,
            source="OFFICIAL",
            native="parallel stain/scanner renderer",
            scanner=True,
            composition=True,
            target=False,
            track="A,B,C,D",
            tier=1,
            kind="OURS / INTERNAL",
            notes="Existing M1 checkpoint; reported honestly if stronger than ordered.",
        ),
        _spec(
            "factorstain",
            "OURS-Ordered",
            "FactorStain ordered stain-to-scanner renderer",
            2026,
            source="OFFICIAL",
            native="ordered stain then scanner renderer",
            scanner=True,
            composition=True,
            target=False,
            track="A,B,C,D",
            tier=1,
            kind="OURS / INTERNAL",
            notes="Existing M1 checkpoint; same data, resolution, validation, and optimization budget as internal baselines.",
        ),
    ]
}


TIER_METHODS = {
    1: [name for name, spec in BASELINES.items() if spec.tier == 1],
    2: [name for name, spec in BASELINES.items() if spec.tier == 2],
    3: [name for name, spec in BASELINES.items() if spec.tier == 3],
}


def get_baseline(name: str) -> BaselineSpec:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "histogram_matching": "histogram",
        "3dlut": "scanner_lut",
        "ours_ordered": "factorstain",
        "ours_parallel": "parallel",
    }
    key = aliases.get(key, key)
    if key not in BASELINES:
        raise KeyError(f"Unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[key]


def resolve_methods(method: str = "", methods: str = "", tier: str = "1") -> list[str]:
    explicit = methods or method
    if explicit:
        resolved = [
            get_baseline(value).method_name
            for value in explicit.split(",")
            if value.strip()
        ]
    elif tier.lower() == "all":
        resolved = list(BASELINES)
    else:
        tier_number = int(tier)
        if tier_number not in TIER_METHODS:
            raise ValueError("TIER must be 1, 2, 3, or all")
        resolved = TIER_METHODS[tier_number]
    return list(dict.fromkeys(resolved))
