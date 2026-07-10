#!/usr/bin/env python3
"""Generate paper-formatted baseline result tables (Section 4.3–4.5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "paper_baseline_tables.md"

# Original paper values (for side-by-side comparison; Section 4.3–4.5).
ORIGINAL_TABLE1 = {
    "30%": {
        "WNNM-MC": (30.24, 0.80, 0.183),
        "TNNR": (33.02, 0.91, 0.114),
        "GUIG": (28.03, 0.73, 0.415),
        "AECF": (30.51, 0.82, 0.224),
        "DMF": (30.29, 0.82, 0.236),
        "MC-ML": (28.72, 0.82, 0.248),
        "Proposed": (31.59, 0.92, 0.090),
    },
    "50%": {
        "WNNM-MC": (25.97, 0.61, 0.417),
        "TNNR": (28.68, 0.77, 0.317),
        "GUIG": (26.45, 0.65, 0.494),
        "AECF": (26.66, 0.66, 0.407),
        "DMF": (26.57, 0.67, 0.430),
        "MC-ML": (25.53, 0.68, 0.455),
        "Proposed": (31.40, 0.90, 0.111),
    },
    "70%": {
        "WNNM-MC": (21.87, 0.36, 0.733),
        "TNNR": (23.91, 0.52, 0.625),
        "GUIG": (14.03, 0.09, 0.870),
        "AECF": (20.65, 0.31, 0.863),
        "DMF": (19.80, 0.31, 0.874),
        "MC-ML": (22.73, 0.52, 0.693),
        "Proposed": (25.54, 0.73, 0.391),
    },
}

ORIGINAL_TABLE2 = {
    "Image-2": {
        "WNNM-MC": (38.56, 0.98, 0.072),
        "TNNR": (17.27, 0.93, 0.429),
        "GUIG": (22.40, 0.91, 0.334),
        "AECF": (26.76, 0.95, 0.234),
        "DMF": (30.59, 0.91, 0.190),
        "MC-ML": (38.87, 0.99, 0.057),
        "Proposed": (41.65, 0.99, 0.022),
    },
    "Image-4": {
        "WNNM-MC": (39.90, 0.98, 0.040),
        "TNNR": (20.47, 0.94, 0.351),
        "GUIG": (22.81, 0.91, 0.369),
        "AECF": (24.49, 0.94, 0.276),
        "DMF": (24.62, 0.85, 0.143),
        "MC-ML": (36.70, 0.98, 0.039),
        "Proposed": (37.20, 0.98, 0.039),
    },
}

ORIGINAL_TABLE3 = {
    "Random": {
        "DMF": (8.59, 31.17, 0.872, 0.261),
        "MC-ML": (16.24, 31.14, 0.861, 0.239),
        "Proposed": (57.53, 37.03, 0.968, 0.034),
    },
    "Block": {
        "DMF": (5.70, 27.97, 0.903, 0.201),
        "MC-ML": (10.14, 22.95, 0.823, 0.376),
        "Proposed": (36.90, 27.97, 0.903, 0.201),
    },
}

ORIGINAL_TABLE4 = {
    "Roundabout (720 × 636)": {
        "DMF": (249.6, 20.62, 0.460, 0.524),
        "MC-ML": (248.8, 20.05, 0.570, 0.490),
        "Proposed": (361.5, 27.14, 0.860, 0.107),
    },
    "Harbor (940 × 559)": {
        "DMF": (304.2, 22.87, 0.700, 0.564),
        "MC-ML": (306.0, 27.73, 0.820, 0.411),
        "Proposed": (372.9, 29.24, 0.870, 0.293),
    },
    "Parking Lot (635 × 657)": {
        "DMF": (235.4, 19.93, 0.430, 0.590),
        "MC-ML": (238.3, 19.86, 0.580, 0.513),
        "Proposed": (356.3, 23.50, 0.730, 0.253),
    },
}

ORIGINAL_METHODS = ["WNNM-MC", "TNNR", "GUIG", "AECF", "DMF", "MC-ML", "Proposed"]
NEW_BASELINES = ["LaMa", "MAT", "RePaint", "DDRM"]
BASELINE_KEY = {"LaMa": "lama", "MAT": "mat", "RePaint": "repaint", "DDRM": "ddrm"}


def _fmt_psnr(v: float) -> str:
    return f"{v:.2f}"


def _fmt_ssim(v: float) -> str:
    return f"{v:.2f}"


def _fmt_lpips(v: float) -> str:
    return f"{v:.3f}"


def _fmt_time(v: float) -> str:
    return f"{v:.2f}"


def _metric_triplet(psnr: float, ssim: float, lpips: float) -> tuple[str, str, str]:
    return _fmt_psnr(psnr), _fmt_ssim(ssim), _fmt_lpips(lpips)


def _load_random_means() -> dict[str, dict[str, tuple[float, float, float]]]:
    df = pd.read_csv(RESULTS_DIR / "random_pixel_masks" / "metrics.csv")
    ratio_map = {"ratio_30": "30%", "ratio_50": "50%", "ratio_70": "70%"}
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for case, ratio in ratio_map.items():
        sub = df[df["case"] == case].groupby("baseline")[["psnr", "ssim", "lpips"]].mean()
        out[ratio] = {
            name: (sub.loc[key, "psnr"], sub.loc[key, "ssim"], sub.loc[key, "lpips"])
            for name, key in BASELINE_KEY.items()
        }
    return out


def _load_block() -> dict[str, dict[str, tuple[float, float, float]]]:
    df = pd.read_csv(RESULTS_DIR / "block_masks" / "metrics.csv")
    case_map = {"image-2": "Image-2", "image-4": "Image-4"}
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for case, label in case_map.items():
        sub = df[df["case"] == case].set_index("baseline")
        out[label] = {
            name: (sub.loc[key, "psnr"], sub.loc[key, "ssim"], sub.loc[key, "lpips"])
            for name, key in BASELINE_KEY.items()
        }
    return out


def _load_missing_patterns() -> dict[str, dict[str, tuple[float, float, float, float]]]:
    df = pd.read_csv(RESULTS_DIR / "missing_patterns" / "metrics.csv")
    case_map = {"random": "Random", "block": "Block"}
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for case, label in case_map.items():
        sub = df[df["case"] == case].set_index("baseline")
        out[label] = {
            name: (
                sub.loc[key, "time_sec"],
                sub.loc[key, "psnr"],
                sub.loc[key, "ssim"],
                sub.loc[key, "lpips"],
            )
            for name, key in BASELINE_KEY.items()
        }
    return out


def _load_scenes() -> dict[str, dict[str, tuple[float, float, float, float]]]:
    df = pd.read_csv(RESULTS_DIR / "scenes_resolutions" / "metrics.csv")
    case_map = {
        "roundabout": "Roundabout (720 × 636)",
        "harbor": "Harbor (940 × 559)",
        "parking_lot": "Parking Lot (635 × 657)",
    }
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for case, label in case_map.items():
        sub = df[df["case"] == case].set_index("baseline")
        out[label] = {
            name: (
                sub.loc[key, "time_sec"],
                sub.loc[key, "psnr"],
                sub.loc[key, "ssim"],
                sub.loc[key, "lpips"],
            )
            for name, key in BASELINE_KEY.items()
        }
    return out


def _table1(random_means: dict) -> list[str]:
    lines = [
        "## Table 1. Random pixel mask of real remote sensing images (Section 4.3.1)",
        "",
        "Image: `leida.png` (900 × 900). Metrics are averaged over **20 trials** per missing rate.",
        "",
        "| Ratios | Indicator | "
        + " | ".join(ORIGINAL_METHODS + NEW_BASELINES)
        + " |",
        "| --- | --- | " + " | ".join(["---"] * (len(ORIGINAL_METHODS) + len(NEW_BASELINES))) + " |",
    ]
    for ratio in ["30%", "50%", "70%"]:
        for idx, metric in enumerate(["PSNR", "SSIM", "LPIPS"]):
            row = [ratio if idx == 0 else "", metric]
            for method in ORIGINAL_METHODS:
                vals = ORIGINAL_TABLE1[ratio][method]
                row.append(_metric_triplet(*vals)[idx])
            for method in NEW_BASELINES:
                vals = random_means[ratio][method]
                row.append(_metric_triplet(*vals)[idx])
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _table2(block: dict) -> list[str]:
    lines = [
        "## Table 2. Experiment with block pixel masks (Section 4.3.2)",
        "",
        "Images: Image-2 / Image-4 (300 × 300).",
        "",
        "| Image | Indicator | "
        + " | ".join(ORIGINAL_METHODS + NEW_BASELINES)
        + " |",
        "| --- | --- | " + " | ".join(["---"] * (len(ORIGINAL_METHODS) + len(NEW_BASELINES))) + " |",
    ]
    for image in ["Image-2", "Image-4"]:
        for idx, metric in enumerate(["PSNR", "SSIM", "LPIPS"]):
            row = [image if idx == 0 else "", metric]
            for method in ORIGINAL_METHODS:
                vals = ORIGINAL_TABLE2[image][method]
                row.append(_metric_triplet(*vals)[idx])
            for method in NEW_BASELINES:
                vals = block[image][method]
                row.append(_metric_triplet(*vals)[idx])
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _table3(missing: dict) -> list[str]:
    lines = [
        "## Table 3. Comparison between random masks and block masks (Section 4.4)",
        "",
        "Image: `leida.png` resized to 900 × 900, missing rate 30%.",
        "",
        "| Missing Patterns | Method | Clock time (s) | PSNR | SSIM | LPIPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pattern in ["Random", "Block"]:
        for method in ["DMF", "MC-ML", "Proposed"]:
            t, p, s, l = ORIGINAL_TABLE3[pattern][method]
            lines.append(
                f"| {pattern} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )
        for method in NEW_BASELINES:
            t, p, s, l = missing[pattern][method]
            lines.append(
                f"| {pattern} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )
    return lines


def _table4(scenes: dict) -> list[str]:
    lines = [
        "## Table 4. Comparison between scenes and image resolutions (Section 4.5)",
        "",
        "30% random pixel mask, seed = 42. Native scene resolutions.",
        "",
        "| Scenes & Resolutions | Method | Clock time (s) | PSNR | SSIM | LPIPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scene in [
        "Roundabout (720 × 636)",
        "Harbor (940 × 559)",
        "Parking Lot (635 × 657)",
    ]:
        for method in ["DMF", "MC-ML", "Proposed"]:
            t, p, s, l = ORIGINAL_TABLE4[scene][method]
            lines.append(
                f"| {scene} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )
        for method in NEW_BASELINES:
            t, p, s, l = scenes[scene][method]
            lines.append(
                f"| {scene} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )
    return lines


def _supplement_only_tables(
    random_means: dict,
    block: dict,
    missing: dict,
    scenes: dict,
) -> list[str]:
    lines = [
        "---",
        "",
        "## Supplement-only tables (4 new baselines, for yellow-highlight columns)",
        "",
        "### Table 1 supplement",
        "",
        "| Ratios | Indicator | LaMa | MAT | RePaint | DDRM |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for ratio in ["30%", "50%", "70%"]:
        for idx, metric in enumerate(["PSNR", "SSIM", "LPIPS"]):
            row = [ratio if idx == 0 else "", metric]
            for method in NEW_BASELINES:
                vals = random_means[ratio][method]
                row.append(_metric_triplet(*vals)[idx])
            lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Table 2 supplement",
        "",
        "| Image | Indicator | LaMa | MAT | RePaint | DDRM |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for image in ["Image-2", "Image-4"]:
        for idx, metric in enumerate(["PSNR", "SSIM", "LPIPS"]):
            row = [image if idx == 0 else "", metric]
            for method in NEW_BASELINES:
                vals = block[image][method]
                row.append(_metric_triplet(*vals)[idx])
            lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Table 3 supplement",
        "",
        "| Missing Patterns | Method | Clock time (s) | PSNR | SSIM | LPIPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pattern in ["Random", "Block"]:
        for method in NEW_BASELINES:
            t, p, s, l = missing[pattern][method]
            lines.append(
                f"| {pattern} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )

    lines += [
        "",
        "### Table 4 supplement",
        "",
        "| Scenes & Resolutions | Method | Clock time (s) | PSNR | SSIM | LPIPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scene in [
        "Roundabout (720 × 636)",
        "Harbor (940 × 559)",
        "Parking Lot (635 × 657)",
    ]:
        for method in NEW_BASELINES:
            t, p, s, l = scenes[scene][method]
            lines.append(
                f"| {scene} | {method} | {_fmt_time(t)} | {_fmt_psnr(p)} | {_fmt_ssim(s)} | {_fmt_lpips(l)} |"
            )
    return lines


def main() -> None:
    random_means = _load_random_means()
    block = _load_block()
    missing = _load_missing_patterns()
    scenes = _load_scenes()

    lines = [
        "# Baseline Supplement Results (Paper Tables)",
        "",
        "Formatted to match the original paper tables in Section 4.3–4.5.",
        "Original-method columns are copied from the submitted manuscript; "
        "**LaMa / MAT / RePaint / DDRM** columns are computed by `baseline-suite`.",
        "",
        "### Notes",
        "",
        "- **Table 1**: 20-trial average per missing rate (30%, 50%, 70%).",
        "- **Table 2**: single run on Image-2 / Image-4 (300 × 300).",
        "- **Table 3 & 4**: include clock time (seconds) on NVIDIA GPU server.",
        "- RePaint / DDRM infer at 256 × 256 then resize back; MAT pads to 512-multiple sizes.",
        "- Hardware for new baselines: GPU server (4090-class), same batch as Section 4.4 timing.",
        "",
    ]
    lines.extend(_table1(random_means))
    lines.append("")
    lines.extend(_table2(block))
    lines.append("")
    lines.extend(_table3(missing))
    lines.append("")
    lines.extend(_table4(scenes))
    lines.append("")
    lines.extend(_supplement_only_tables(random_means, block, missing, scenes))
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
