"""
Visualize sensitivity ablation results.

Reads ablation_sensitivity_results.csv and generates:
  Figure 1 (per PDE): Sensitive params vs num_batches (lines = maxout, subplots = var_threshold)
  Figure 2 (per PDE): Sensitive params vs max_outputs  (lines = nbatch, subplots = var_threshold)
  Figure 3: Heatmaps of sens-pde-all (subplots = var_threshold)
  Figure 4: Per-PDE heatmaps side-by-side for each var_threshold

Usage:
    python plot_ablation.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------- configuration ---------------
CSV_PATH = "ablation_sensitivity_results.csv"
OUT_DIR = "ablation_plots"
PDE_LABELS = {
    "sens-pde-0": "Allen-Cahn",
    "sens-pde-1": "Nagumo",
    "sens-pde-2": "Wave",
    "sens-pde-all": "Combined",
}
# ----------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ── Read CSV with stdlib ──────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Detect PDE columns dynamically
SENS_COLS = [k for k in rows[0] if k.startswith("sens-pde-")]
# Ensure "sens-pde-all" is last
SENS_COLS = sorted([c for c in SENS_COLS if c != "sens-pde-all"]) + ["sens-pde-all"]

# Parse into typed lists
data = {col: [] for col in ["nbatch", "maxout", "var_threshold"] + SENS_COLS}
for r in rows:
    data["nbatch"].append(int(r["nbatch"]))
    data["maxout"].append(int(r["maxout"]))
    data["var_threshold"].append(float(r["var_threshold"]))
    for col in SENS_COLS:
        data[col].append(int(r[col]))

# Convert to numpy arrays
for k in data:
    data[k] = np.array(data[k])

var_thresholds = sorted(set(data["var_threshold"]))
maxouts = sorted(set(data["maxout"]))
nbatches = sorted(set(data["nbatch"]))


# ── Helper: boolean mask ──────────────────────────────────────────────────────
def mask_eq(arr, val):
    if isinstance(val, float):
        return np.isclose(arr, val)
    return arr == val


# ── Colour palettes ──────────────────────────────────────────────────────────
cmap_mo = plt.cm.viridis
cmap_nb = plt.cm.plasma

def colour_map(values, cmap):
    n = len(values)
    return {v: cmap(i / max(n - 1, 1)) for i, v in enumerate(values)}

colours_mo = colour_map(maxouts, cmap_mo)
colours_nb = colour_map(nbatches, cmap_nb)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: sens vs num_batches  (subplots = var_threshold, lines = maxout)
# ═══════════════════════════════════════════════════════════════════════════════
for col in SENS_COLS:
    fig, axes = plt.subplots(1, len(var_thresholds),
                             figsize=(6 * len(var_thresholds), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, vt in zip(axes, var_thresholds):
        m_vt = mask_eq(data["var_threshold"], vt)
        for mo in maxouts:
            m = m_vt & mask_eq(data["maxout"], mo)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            order = np.argsort(data["nbatch"][idx])
            xvals = data["nbatch"][idx][order]
            yvals = data[col][idx][order]
            ax.plot(xvals, yvals, "o-", color=colours_mo[mo],
                    label=f"maxout={mo}", linewidth=1.5, markersize=4)

        ax.set_title(f"var_threshold = {vt}", fontsize=12)
        ax.set_xlabel("num_batches", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(nbatches)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("# Sensitive Parameters", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"{PDE_LABELS.get(col, col)}:  Sensitive params vs num_batches",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"sens_vs_nbatch_{col}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: sens vs max_outputs  (subplots = var_threshold, lines = nbatch)
# ═══════════════════════════════════════════════════════════════════════════════
for col in SENS_COLS:
    fig, axes = plt.subplots(1, len(var_thresholds),
                             figsize=(6 * len(var_thresholds), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, vt in zip(axes, var_thresholds):
        m_vt = mask_eq(data["var_threshold"], vt)
        for nb in nbatches:
            m = m_vt & mask_eq(data["nbatch"], nb)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            order = np.argsort(data["maxout"][idx])
            xvals = data["maxout"][idx][order]
            yvals = data[col][idx][order]
            ax.plot(xvals, yvals, "s-", color=colours_nb[nb],
                    label=f"nbatch={nb}", linewidth=1.5, markersize=4)

        ax.set_title(f"var_threshold = {vt}", fontsize=12)
        ax.set_xlabel("max_outputs", fontsize=11)
        ax.set_xscale("log", base=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(maxouts)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("# Sensitive Parameters", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"{PDE_LABELS.get(col, col)}:  Sensitive params vs max_outputs",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"sens_vs_maxout_{col}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Heatmap of sens-pde-all (subplots = var_threshold)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(var_thresholds),
                         figsize=(6 * len(var_thresholds), 5), squeeze=False)
axes = axes[0]

for ax, vt in zip(axes, var_thresholds):
    m_vt = mask_eq(data["var_threshold"], vt)
    matrix = np.full((len(nbatches), len(maxouts)), np.nan)
    for i, nb in enumerate(nbatches):
        for j, mo in enumerate(maxouts):
            m = m_vt & mask_eq(data["nbatch"], nb) & mask_eq(data["maxout"], mo)
            idx = np.where(m)[0]
            if len(idx) == 1:
                matrix[i, j] = data["sens-pde-all"][idx[0]]

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(maxouts)))
    ax.set_xticklabels(maxouts, fontsize=9)
    ax.set_yticks(range(len(nbatches)))
    ax.set_yticklabels(nbatches, fontsize=9)
    ax.set_xlabel("max_outputs", fontsize=11)
    ax.set_ylabel("num_batches", fontsize=11)
    ax.set_title(f"var_threshold = {vt}", fontsize=12)

    med = np.nanmedian(matrix)
    for i in range(len(nbatches)):
        for j in range(len(maxouts)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=7,
                        color="white" if v > med else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="# Sensitive Params")

fig.suptitle("Combined (all PDEs):  Heatmap of sensitive parameters",
             fontsize=14, y=1.02)
fig.tight_layout()
fname = os.path.join(OUT_DIR, "heatmap_sens_pde_all.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Per-PDE heatmaps side-by-side for each var_threshold
# ═══════════════════════════════════════════════════════════════════════════════
for vt in var_thresholds:
    m_vt = mask_eq(data["var_threshold"], vt)
    fig, axes = plt.subplots(1, len(SENS_COLS),
                             figsize=(5.5 * len(SENS_COLS), 5), squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, SENS_COLS):
        matrix = np.full((len(nbatches), len(maxouts)), np.nan)
        for i, nb in enumerate(nbatches):
            for j, mo in enumerate(maxouts):
                m = m_vt & mask_eq(data["nbatch"], nb) & mask_eq(data["maxout"], mo)
                idx = np.where(m)[0]
                if len(idx) == 1:
                    matrix[i, j] = data[col][idx[0]]

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(len(maxouts)))
        ax.set_xticklabels(maxouts, fontsize=9)
        ax.set_yticks(range(len(nbatches)))
        ax.set_yticklabels(nbatches, fontsize=9)
        ax.set_xlabel("max_outputs", fontsize=11)
        ax.set_ylabel("num_batches", fontsize=11)
        ax.set_title(PDE_LABELS.get(col, col), fontsize=12)

        med = np.nanmedian(matrix)
        for i in range(len(nbatches)):
            for j in range(len(maxouts)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=7,
                            color="white" if v > med else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Sensitive parameters  (var_threshold = {vt})", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"heatmap_all_pdes_vt{vt}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


print(f"\nAll plots saved to {OUT_DIR}/")
