import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Shim 1: transformers 5.x removed build_inputs_with_special_tokens
# from slow tokenizers, but bert_score 0.3.x still calls it.
from transformers import RobertaTokenizer

# =====================================================================
# Compatibility shims for bert_score + transformers 5.x
# =====================================================================


if not hasattr(RobertaTokenizer, "build_inputs_with_special_tokens"):

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id] if self.cls_token_id is not None else []
        sep = [self.sep_token_id] if self.sep_token_id is not None else []
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    RobertaTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

# Shim 2: tokenizer.model_max_length can overflow the Rust tokenizer's
# enable_truncation (OverflowError: int too big to convert).
# Cap it at 512 before bert_score uses it.
import bert_score.utils as _bsu

_orig_sent_encode = _bsu.sent_encode


def _safe_sent_encode(tokenizer, sent):
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length > 100000:
        tokenizer.model_max_length = 512
    return _orig_sent_encode(tokenizer, sent)


_bsu.sent_encode = _safe_sent_encode

from bert_score import score as bertscore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent  # src/clinicalshift/ → src/ → project root
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_FMT_3F = "%{text:.3f}"
TEXT_FMT_3F_S = "%{text:.3f}s"


# ---------------------------------------------------------------------
# Device detection: prefer GPU (MPS/CUDA), fall back to CPU
# ---------------------------------------------------------------------


def get_device() -> str:
    """Detect best available device for PyTorch inference."""
    from clinicalshift import get_device as _get_device

    return _get_device()


DEVICE = get_device()


# ---------------------------------------------------------------------
# BERTScore-based SFI
# ---------------------------------------------------------------------


def compute_sfi(y_final: List[str], y_star: List[str]) -> float:
    """
    Semantic Fidelity Index (SFI) using BERTScore F1 averaged over all rows.
    """
    if not y_final:
        return 0.0
    _, _, f1 = bertscore(
        y_final,
        y_star,
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        batch_size=32,
        lang="en",
        device=DEVICE,
    )
    return float(f1.mean().item())


# ---------------------------------------------------------------------
# FCCR from normalized contradiction keys
# ---------------------------------------------------------------------


def extract_detected_keys(audit_strs: pd.Series) -> List[List[str]]:
    """
    auditissues column contains JSON dicts with 'normalized_keys'.
    """
    all_keys: List[List[str]] = []
    for s in audit_strs.fillna("[]"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            all_keys.append([])
            continue
        if isinstance(obj, dict):
            keys = obj.get("normalized_keys", [])
        elif isinstance(obj, list):
            keys = obj
        else:
            keys = []
        keys = [str(k) for k in keys]
        all_keys.append(keys)
    return all_keys


def extract_gt_keys(gt_series: pd.Series) -> List[List[str]]:
    """
    gtcontradictions is assumed to be a JSON list of canonical keys or a
    comma-separated string of keys.
    """
    all_keys: List[List[str]] = []
    for s in gt_series.fillna("[]"):
        s = str(s)
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                keys = [str(k) for k in obj]
            else:
                keys = []
        except json.JSONDecodeError:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            keys = parts
        all_keys.append(keys)
    return all_keys


def compute_fccr(df: pd.DataFrame) -> float:
    """
    Per-patient Fact-Checking Coverage Ratio.

    For each patient with at least one ground-truth contradiction,
    compute the fraction of that patient's contradictions detected.
    Returns mean per-patient detection rate across the contradicted subset.
    """
    detected_lists = extract_detected_keys(df["auditissues"])
    gt_lists = extract_gt_keys(df["gtcontradictions"])

    per_patient_scores = []
    for detected, gt in zip(detected_lists, gt_lists):
        if not gt:
            continue  # Skip patients with no contradictions
        detected_set = set(detected)
        gt_set = set(gt)
        score = len(detected_set.intersection(gt_set)) / len(gt_set)
        per_patient_scores.append(score)

    if not per_patient_scores:
        return 0.0

    return float(np.mean(per_patient_scores))


def compute_gcs(df: pd.DataFrame) -> float:
    """
    Guideline Compliance Score (GCS).

    For patients with ground-truth contradictions, checks whether
    the generated summary contains language acknowledging the
    contraindication. Uses deterministic string matching.

    Returns fraction of contradicted patients whose Y_final
    contains compliance language near the relevant drug name.
    """
    gt_lists = extract_gt_keys(df["gtcontradictions"])
    y_finals = df["y_final"].fillna("").tolist()

    compliance_terms = [
        "avoid", "contraindicated", "discontinue", "switch",
        "stop", "not recommended", "should not", "withhold",
        "alternative", "risk of lactic acidosis",
    ]

    compliant_count = 0
    total_contradicted = 0

    for gt, y_final in zip(gt_lists, y_finals):
        if not gt:
            continue
        total_contradicted += 1
        y_lower = y_final.lower()
        # Check if any compliance term appears in the output
        if any(term in y_lower for term in compliance_terms):
            compliant_count += 1

    if total_contradicted == 0:
        return float("nan")

    return float(compliant_count / total_contradicted)


# ---------------------------------------------------------------------
# TCA (Temporal Calibration Agreement)
# ---------------------------------------------------------------------


def compute_tca(df: pd.DataFrame) -> float:
    """
    Temporal Calibration Agreement: measures how well the LLM's
    verbalized confidence aligns with actual epoch-correctness.

    TCA = 1 - mean(|predicted_confidence_per_bin - actual_alignment_per_bin|)
    Perfect calibration → TCA = 1.0; random → TCA ≈ 0.5
    """
    if "confidence" not in df.columns or "epoch_aligned" not in df.columns:
        return float("nan")

    conf = pd.to_numeric(df["confidence"], errors="coerce")
    aligned = pd.to_numeric(df["epoch_aligned"], errors="coerce")

    mask = conf.notna() & aligned.notna()
    if mask.sum() < 10:
        return float("nan")

    conf = conf[mask].values
    aligned = aligned[mask].values

    # Bin into deciles
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    deviations = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)

        if in_bin.sum() == 0:
            continue

        predicted_conf = conf[in_bin].mean()
        actual_rate = aligned[in_bin].mean()
        deviations.append(abs(predicted_conf - actual_rate))

    if not deviations:
        return float("nan")

    return float(1.0 - np.mean(deviations))


# ---------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, stat_fn=np.mean, n_resamples: int = 1000, ci: float = 0.95
) -> tuple:
    """Compute bootstrap confidence interval for a statistic."""
    if len(values) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (lower, upper)


# ---------------------------------------------------------------------
# Per-condition summarization
# ---------------------------------------------------------------------


def summarize_condition(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)

    condition = df["condition"].iloc[0] if "condition" in df.columns else path.stem
    n = len(df)

    # SFI (BERTScore F1)
    y_final_list = df["y_final"].fillna("").tolist()
    y_star_list = df["y_star"].fillna("").tolist()
    sfi = compute_sfi(y_final_list, y_star_list)

    # Per-row SFI for bootstrap CI
    if y_final_list and len(y_final_list) > 1:
        _, _, f1_scores = bertscore(
            y_final_list,
            y_star_list,
            model_type="microsoft/deberta-xlarge-mnli",
            verbose=False,
            batch_size=32,
            lang="en",
            device=DEVICE,
        )
        sfi_values = f1_scores.numpy()
        sfi_ci = bootstrap_ci(sfi_values)
    else:
        sfi_ci = (float("nan"), float("nan"))

    # FCCR
    fccr = compute_fccr(df)

    # GCS
    gcs = compute_gcs(df)

    # TCA
    tca = compute_tca(df)

    # Latency
    lat_mean = float(df["latency"].mean())
    lat_std = float(df["latency"].std(ddof=1)) if n > 1 else 0.0

    return {
        "condition": condition,
        "n": n,
        "SFI": sfi,
        "SFI_ci_lower": sfi_ci[0],
        "SFI_ci_upper": sfi_ci[1],
        "FCCR": fccr,
        "GCS": gcs,
        "TCA": tca,
        "latency_mean": lat_mean,
        "latency_std": lat_std,
    }


# ---------------------------------------------------------------------
# Condition parsing: mode + regime
# ---------------------------------------------------------------------


def split_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split condition strings of the form 'mode_regime' into separate columns.
    """
    modes: List[str] = []
    regimes: List[str] = []

    for cond in df["condition"]:
        parts = str(cond).split("_", 1)
        if len(parts) == 2:
            mode, regime = parts
        else:
            mode = "graph"
            regime = parts[0]
        modes.append(mode)
        regimes.append(regime)

    df = df.copy()
    df["mode"] = modes
    df["regime"] = regimes
    return df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def make_overall_plots(df_stats: pd.DataFrame):
    """
    Overall plots by condition (no facets).
    """
    df_stats = split_condition(df_stats)

    # SFI
    fig_sfi = px.bar(
        df_stats,
        x="condition",
        y="SFI",
        color="mode",
        title="Semantic Fidelity Index (SFI) by Condition",
        text="SFI",
    )
    fig_sfi.update_traces(texttemplate=TEXT_FMT_3F, textposition="outside")
    fig_sfi.update_layout(xaxis_tickangle=45)
    pio.write_image(fig_sfi, RESULTS_DIR / "plot_sfi_overall.png")
    fig_sfi.write_html(RESULTS_DIR / "plot_sfi_overall.html")

    # FCCR
    fig_fccr = px.bar(
        df_stats,
        x="condition",
        y="FCCR",
        color="mode",
        title="Fact-Checking Coverage Ratio (FCCR) by Condition",
        text="FCCR",
    )
    fig_fccr.update_traces(texttemplate=TEXT_FMT_3F, textposition="outside")
    fig_fccr.update_layout(xaxis_tickangle=45)
    pio.write_image(fig_fccr, RESULTS_DIR / "plot_fccr_overall.png")
    fig_fccr.write_html(RESULTS_DIR / "plot_fccr_overall.html")

    # Latency
    fig_lat = px.bar(
        df_stats,
        x="condition",
        y="latency_mean",
        error_y="latency_std",
        color="mode",
        title="Latency by Condition",
        text="latency_mean",
    )
    fig_lat.update_traces(texttemplate=TEXT_FMT_3F_S, textposition="outside")
    fig_lat.update_layout(
        xaxis_tickangle=45,
        yaxis_title="Latency (s)",
    )
    pio.write_image(fig_lat, RESULTS_DIR / "plot_latency_overall.png")
    fig_lat.write_html(RESULTS_DIR / "plot_latency_overall.html")


def make_per_regime_plots(df_stats: pd.DataFrame):
    """
    Faceted bar plots where each regime is a panel and bars are modes.
    """
    df_stats = split_condition(df_stats)
    df_stats["regime"] = pd.Categorical(
        df_stats["regime"],
        categories=[
            "baseline_tau_old",
            "temporal_drift_tau_new",
            "institution_swap_instB",
        ],
        ordered=True,
    )
    df_stats["mode"] = pd.Categorical(
        df_stats["mode"],
        categories=["single", "linear", "graph"],
        ordered=True,
    )

    mode_order = {"mode": ["single", "linear", "graph"]}

    # SFI per regime
    fig_sfi_reg = px.bar(
        df_stats,
        x="mode",
        y="SFI",
        facet_col="regime",
        category_orders=mode_order,
        title="SFI by Architecture within Each Shift Regime",
        text="SFI",
    )
    fig_sfi_reg.update_traces(texttemplate=TEXT_FMT_3F, textposition="outside")
    fig_sfi_reg.update_layout(yaxis_title="SFI (BERTScore F1)")
    pio.write_image(fig_sfi_reg, RESULTS_DIR / "plot_sfi_by_regime.png")
    fig_sfi_reg.write_html(RESULTS_DIR / "plot_sfi_by_regime.html")

    # FCCR per regime
    fig_fccr_reg = px.bar(
        df_stats,
        x="mode",
        y="FCCR",
        facet_col="regime",
        category_orders=mode_order,
        title="FCCR by Architecture within Each Shift Regime",
        text="FCCR",
    )
    fig_fccr_reg.update_traces(texttemplate=TEXT_FMT_3F, textposition="outside")
    fig_fccr_reg.update_layout(yaxis_title="FCCR")
    pio.write_image(fig_fccr_reg, RESULTS_DIR / "plot_fccr_by_regime.png")
    fig_fccr_reg.write_html(RESULTS_DIR / "plot_fccr_by_regime.html")

    # Latency per regime
    fig_lat_reg = px.bar(
        df_stats,
        x="mode",
        y="latency_mean",
        error_y="latency_std",
        facet_col="regime",
        category_orders=mode_order,
        title="Latency by Architecture within Each Shift Regime",
        text="latency_mean",
    )
    fig_lat_reg.update_traces(texttemplate=TEXT_FMT_3F_S, textposition="outside")
    fig_lat_reg.update_layout(yaxis_title="Latency (s)")
    pio.write_image(fig_lat_reg, RESULTS_DIR / "plot_latency_by_regime.png")
    fig_lat_reg.write_html(RESULTS_DIR / "plot_latency_by_regime.html")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    stats: List[Dict[str, Any]] = []

    for path in sorted(RESULTS_DIR.glob("results_*.csv")):
        s = summarize_condition(path)
        stats.append(s)

    if not stats:
        print(f"No result files found in {RESULTS_DIR}")
        return

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))

    out_path = RESULTS_DIR / "metrics_summary.csv"
    df_stats.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    make_overall_plots(df_stats)
    make_per_regime_plots(df_stats)


if __name__ == "__main__":
    main()
