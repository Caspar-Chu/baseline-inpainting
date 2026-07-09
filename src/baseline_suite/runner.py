from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from baseline_suite.config import ExperimentConfig, load_config
from baseline_suite.experiments import Sample, build_samples, cache_masks
from baseline_suite.io import save_image
from baseline_suite.metrics import MetricResult, evaluate
from baseline_suite.paths import ensure_dirs
from baseline_suite.registry import create_baseline


def _sample_id(sample: Sample) -> str:
    parts = [sample.experiment, sample.case_name]
    if sample.trial is not None:
        parts.append(f"trial{sample.trial:02d}")
    return "__".join(parts)


def run_experiment(
    experiment: str,
    baselines: list[str] | None = None,
    *,
    config_path: str | Path | None = None,
    cache_only: bool = False,
) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config(config_path)
    method_names = baselines or cfg.baselines
    samples = build_samples(cfg, experiment)

    if cache_only:
        cache_masks(cfg, samples)
        return pd.DataFrame()

    rows: list[dict] = []
    for sample in tqdm(samples, desc=experiment):
        cache_masks(cfg, [sample])
        metrics_list = cfg.experiments[experiment].get("metrics", ["psnr", "ssim", "lpips"])

        for name in method_names:
            inpainter = create_baseline(name)
            try:
                result = inpainter.inpaint(
                    sample.masked_image,
                    sample.keep_mask,
                    reference=sample.image,
                )
                restored = result.image
                metrics: MetricResult = evaluate(
                    sample.image,
                    restored,
                    metrics=metrics_list,
                )
                if "time_sec" in metrics_list:
                    metrics.time_sec = result.time_sec

                out_dir = (
                    cfg.output_root
                    / experiment
                    / sample.case_name
                    / name
                )
                if sample.trial is not None:
                    out_dir = out_dir / f"trial_{sample.trial:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                save_image(out_dir / "restored.png", restored)

                row = {
                    "experiment": experiment,
                    "case": sample.case_name,
                    "baseline": name,
                    "trial": sample.trial,
                    **sample.meta,
                    **metrics.as_dict(),
                }
                rows.append(row)
            except NotImplementedError as exc:
                rows.append(
                    {
                        "experiment": experiment,
                        "case": sample.case_name,
                        "baseline": name,
                        "trial": sample.trial,
                        **sample.meta,
                        "status": "not_implemented",
                        "error": str(exc),
                    }
                )
            except FileNotFoundError as exc:
                rows.append(
                    {
                        "experiment": experiment,
                        "case": sample.case_name,
                        "baseline": name,
                        "trial": sample.trial,
                        **sample.meta,
                        "status": "missing_repo",
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = cfg.output_root / experiment / "metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and method_names:
        existing = pd.read_csv(out_csv)
        existing = existing[~existing["baseline"].isin(method_names)]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    summary_path = cfg.output_root / experiment / "summary.json"
    baseline_names = sorted(df["baseline"].dropna().unique().tolist()) if "baseline" in df.columns else method_names
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment,
                "num_samples": len(samples),
                "baselines": baseline_names,
                "rows": len(df),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return df


def aggregate_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across trials for experiments like 4.3.1."""
    metric_cols = [c for c in ["psnr", "ssim", "lpips", "time_sec"] if c in df.columns]
    if "trial" not in df.columns or df["trial"].isna().all():
        return df
    group_cols = ["experiment", "case", "baseline"]
    extra = [c for c in df.columns if c not in group_cols + metric_cols + ["trial", "status", "error"]]
    grouped = df.groupby(group_cols + extra, dropna=False)[metric_cols].mean().reset_index()
    return grouped
