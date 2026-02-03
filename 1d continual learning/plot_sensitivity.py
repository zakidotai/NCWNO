#!/usr/bin/env python3
"""
Plot sensitivity analysis results saved by sens.py or sens_ncwno_data.py.

Generates paper-ready visualizations:
1) Cumulative variance vs number of parameters (sorted by sensitivity)
2) Histogram of sensitivities (log10 scale)
3) Rank vs sensitivity (log-log)
4) Lorenz-style curve: cumulative variance vs fraction of parameters
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_sensitivity(path: str) -> Dict[str, Any]:
    # PyTorch 2.6 defaults weights_only=True; these results are full dicts.
    return torch.load(path, map_location="cpu", weights_only=False)


def select_sensitivity(data: Dict[str, Any], pde_index: int | None) -> Dict[str, Any]:
    if "sensitivity_scores" in data:
        return {
            "sensitivity_scores": data["sensitivity_scores"],
            "var_threshold": data.get("var_threshold"),
            "num_sensitive": data.get("num_sensitive"),
        }

    if "combined_sensitivity" in data:
        if pde_index is None:
            num_sensitive = data.get("num_sensitive")
            combined_analysis = data.get("combined_analysis", {})
            if combined_analysis:
                num_sensitive = combined_analysis.get("num_sensitive_params", num_sensitive)
            return {
                "sensitivity_scores": data["combined_sensitivity"],
                "var_threshold": data.get("var_threshold"),
                "num_sensitive": num_sensitive,
            }

        per_pde = data.get("per_pde_sensitivity")
        if per_pde is None:
            raise KeyError("Missing key 'per_pde_sensitivity' for PDE-specific plot.")
        if pde_index < 0 or pde_index >= len(per_pde):
            raise IndexError(f"pde_index {pde_index} out of range (0..{len(per_pde)-1})")
        analysis_results = data.get("analysis_results", [])
        num_sensitive = None
        if pde_index < len(analysis_results):
            num_sensitive = analysis_results[pde_index].get("num_sensitive_params")
        return {
            "sensitivity_scores": per_pde[pde_index],
            "var_threshold": data.get("var_threshold"),
            "num_sensitive": num_sensitive,
        }

    raise KeyError(
        "Unsupported sensitivity file. Expected keys from sens.py "
        "('sensitivity_scores') or sens_ncwno_data.py ('combined_sensitivity')."
    )


def _pde_label(data: Dict[str, Any], idx: int) -> str:
    data_paths = data.get("data_paths", None)
    if data_paths and idx < len(data_paths):
        return os.path.basename(str(data_paths[idx]))
    return f"PDE {idx}"


def plot_all_pdes_cumulative(data: Dict[str, Any], outdir: str, title: str, dpi: int) -> None:
    per_pde = data.get("per_pde_sensitivity")
    if per_pde is None:
        raise KeyError("Missing key 'per_pde_sensitivity' for --plot_all_pdes.")

    fig = plt.figure(figsize=(7, 4))
    for idx, sens_tensor in enumerate(per_pde):
        sens = sens_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        total = sens.sum()
        if total <= 0:
            total = 1.0
        sorted_sens = np.sort(sens)[::-1]
        cumsum = np.cumsum(sorted_sens) / total
        ranks = np.arange(1, len(sorted_sens) + 1)

        plt.plot(ranks, cumsum, lw=1.5, label=_pde_label(data, idx))

    plt.xlabel("Number of parameters (sorted by sensitivity)")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"{title}: Cumulative Variance (All PDEs)")
    plt.legend()
    save_figure(fig, outdir, "cumulative_variance_vs_params_all_pdes", dpi)


def plot_all_pdes_histogram(data: Dict[str, Any], outdir: str, title: str, dpi: int, eps: float) -> None:
    per_pde = data.get("per_pde_sensitivity")
    if per_pde is None:
        raise KeyError("Missing key 'per_pde_sensitivity' for --plot_all_pdes.")

    log_values = []
    for sens_tensor in per_pde:
        sens = sens_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        nonzero = sens[sens > 0]
        if nonzero.size == 0:
            nonzero = sens + eps
        log_values.append(np.log10(nonzero + eps))

    all_vals = np.concatenate(log_values)
    if all_vals.size == 0:
        return

    bins = np.linspace(all_vals.min(), all_vals.max(), 60)
    fig = plt.figure(figsize=(7, 4))
    for idx, vals in enumerate(log_values):
        hist, edges = np.histogram(vals, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, lw=1.5, label=_pde_label(data, idx))

    plt.xlabel("log10(sensitivity)")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title(f"{title}: Sensitivity Histogram (All PDEs)")
    plt.legend()
    save_figure(fig, outdir, "histogram_log_sensitivity_all_pdes", dpi)


def plot_all_pdes_rank(data: Dict[str, Any], outdir: str, title: str, dpi: int, eps: float) -> None:
    per_pde = data.get("per_pde_sensitivity")
    if per_pde is None:
        raise KeyError("Missing key 'per_pde_sensitivity' for --plot_all_pdes.")

    fig = plt.figure(figsize=(7, 4))
    for idx, sens_tensor in enumerate(per_pde):
        sens = sens_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        sorted_sens = np.sort(sens)[::-1]
        ranks = np.arange(1, len(sorted_sens) + 1)
        y = sorted_sens.copy()
        y[y <= 0] = eps
        plt.loglog(ranks, y, lw=1.2, label=_pde_label(data, idx))

    plt.xlabel("Parameter rank (1 = most sensitive)")
    plt.ylabel("Sensitivity")
    plt.title(f"{title}: Rank vs Sensitivity (All PDEs)")
    plt.legend()
    save_figure(fig, outdir, "rank_vs_sensitivity_loglog_all_pdes", dpi)


def plot_all_pdes_lorenz(data: Dict[str, Any], outdir: str, title: str, dpi: int) -> None:
    per_pde = data.get("per_pde_sensitivity")
    if per_pde is None:
        raise KeyError("Missing key 'per_pde_sensitivity' for --plot_all_pdes.")

    fig = plt.figure(figsize=(7, 4))
    for idx, sens_tensor in enumerate(per_pde):
        sens = sens_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        total = sens.sum()
        if total <= 0:
            total = 1.0
        sorted_sens = np.sort(sens)[::-1]
        cumsum = np.cumsum(sorted_sens) / total
        frac_params = np.arange(1, len(sorted_sens) + 1) / len(sorted_sens)
        plt.plot(frac_params, cumsum, lw=1.5, label=_pde_label(data, idx))

    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Uniform")
    plt.xlabel("Fraction of parameters")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"{title}: Sensitivity Concentration (All PDEs)")
    plt.legend()
    save_figure(fig, outdir, "lorenz_curve_all_pdes", dpi)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_figure(fig: plt.Figure, outdir: str, name: str, dpi: int) -> None:
    png_path = os.path.join(outdir, f"{name}.png")
    pdf_path = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sensitivity analysis results.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .pt saved by sens.py or sens_ncwno_data.py",
    )
    parser.add_argument(
        "--pde_index",
        type=int,
        default=None,
        help="If using sens_ncwno_data.py outputs, plot a specific PDE index.",
    )
    parser.add_argument(
        "--plot_all_pdes",
        action="store_true",
        help="Also plot cumulative variance vs params for all PDEs together.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="sensitivity_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sensitivity Analysis",
        help="Base title for plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PNG outputs",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small value for log plots",
    )
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    data = load_sensitivity(args.input)
    selected = select_sensitivity(data, args.pde_index)

    if args.plot_all_pdes and "per_pde_sensitivity" in data:
        plot_all_pdes_cumulative(data, args.outdir, args.title, args.dpi)
        plot_all_pdes_histogram(data, args.outdir, args.title, args.dpi, args.eps)
        plot_all_pdes_rank(data, args.outdir, args.title, args.dpi, args.eps)
        plot_all_pdes_lorenz(data, args.outdir, args.title, args.dpi)

    sens = selected["sensitivity_scores"].detach().cpu().numpy().astype(np.float64)
    sens = sens.reshape(-1)
    total = sens.sum()
    if total <= 0:
        print("Warning: total sensitivity is zero or negative. Plots will be degenerate.")
        total = 1.0

    sorted_sens = np.sort(sens)[::-1]
    cumsum = np.cumsum(sorted_sens) / total
    n_params = len(sorted_sens)
    ranks = np.arange(1, n_params + 1)
    frac_params = ranks / n_params

    var_threshold = selected.get("var_threshold", None)
    num_sensitive = selected.get("num_sensitive", None)

    # 1) Cumulative variance vs number of parameters
    fig = plt.figure(figsize=(7, 4))
    plt.plot(ranks, cumsum, lw=2)
    plt.xlabel("Number of parameters (sorted by sensitivity)")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"{args.title}: Cumulative Variance")
    if var_threshold is not None:
        plt.axhline(var_threshold, color="r", ls="--", lw=1, label="Threshold")
    if num_sensitive is not None:
        plt.axvline(num_sensitive, color="g", ls="--", lw=1, label="Num sensitive")
    if var_threshold is not None or num_sensitive is not None:
        plt.legend()
    save_figure(fig, args.outdir, "cumulative_variance_vs_params", args.dpi)

    # 2) Histogram of sensitivities (log10)
    fig = plt.figure(figsize=(7, 4))
    nonzero = sens[sens > 0]
    if nonzero.size == 0:
        print("Warning: all sensitivities are zero.")
        nonzero = sens + args.eps
    log_sens = np.log10(nonzero + args.eps)
    plt.hist(log_sens, bins=60, color="steelblue", edgecolor="black", alpha=0.85)
    plt.xlabel("log10(sensitivity)")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title(f"{args.title}: Sensitivity Histogram (log10)")
    zero_count = int((sens == 0).sum())
    if zero_count > 0:
        plt.text(0.02, 0.95, f"Zero sensitivities: {zero_count}",
                 transform=plt.gca().transAxes, va="top")
    save_figure(fig, args.outdir, "histogram_log_sensitivity", args.dpi)

    # 3) Rank plot (log-log)
    fig = plt.figure(figsize=(7, 4))
    y = sorted_sens.copy()
    y[y <= 0] = args.eps
    plt.loglog(ranks, y, lw=1.5)
    plt.xlabel("Parameter rank (1 = most sensitive)")
    plt.ylabel("Sensitivity")
    plt.title(f"{args.title}: Rank vs Sensitivity (log-log)")
    save_figure(fig, args.outdir, "rank_vs_sensitivity_loglog", args.dpi)

    # 4) Lorenz-style curve
    fig = plt.figure(figsize=(7, 4))
    plt.plot(frac_params, cumsum, lw=2, label="Sensitivity Lorenz curve")
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Uniform")
    plt.xlabel("Fraction of parameters")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"{args.title}: Sensitivity Concentration")
    plt.legend()
    save_figure(fig, args.outdir, "lorenz_curve", args.dpi)

    print(f"Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()
