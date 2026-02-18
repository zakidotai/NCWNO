"""
Post-process sensitivity ablation results into a CSV file.

Reads all .pt files from results_sensitivity/ and produces a CSV with columns:
  nbatch, maxout, var_threshold, sens-pde-0, sens-pde-1, sens-pde-2, sens-pde-all

Usage:
    python ablation_csv.py
"""

import os
import re
import csv
import torch

# --------------- configuration ---------------
RESULTS_DIR = "results_sensitivity"
OUTPUT_CSV = "ablation_sensitivity_results.csv"
VAR_THRESHOLDS = [ 0.8, 0.9, 0.95]
# ----------------------------------------------

# Pattern: sensitivity_scores_ncwno_maxoutputs_<maxout>_nbatches_<nbatch>.pt
FILENAME_PATTERN = re.compile(
    r"sensitivity_scores_ncwno_maxoutputs_(\w+)_nbatches_(\d+)\.pt"
)


def num_sensitive_at_threshold(cumulative_var_frac, var_threshold):
    """Return number of parameters explaining var_threshold of total variance."""
    return int(torch.sum(cumulative_var_frac <= var_threshold).item()) + 1


def process_file(filepath, maxout, nbatch):
    """Load one .pt file and return rows for every var_threshold."""
    data = torch.load(filepath, map_location="cpu", weights_only=False)

    rows = []
    for vt in VAR_THRESHOLDS:
        row = {
            "nbatch": nbatch,
            "maxout": maxout,
            "var_threshold": vt,
        }

        # --- Combined (averaged across all PDEs) ---
        if "cumulative_variance_fraction" in data:
            row["sens-pde-all"] = num_sensitive_at_threshold(
                data["cumulative_variance_fraction"], vt
            )
        else:
            # Fallback: recompute from raw scores
            cs = data["combined_sensitivity"]
            total = cs.sum()
            sorted_s, _ = torch.sort(cs, descending=True)
            cum = torch.cumsum(sorted_s, 0) / total
            row["sens-pde-all"] = num_sensitive_at_threshold(cum, vt)

        # --- Per-PDE ---
        num_pdes = data.get("num_pdes", 0)
        if "per_pde_sorted_data" in data:
            for pde_idx in range(num_pdes):
                pde_data = data["per_pde_sorted_data"][pde_idx]
                row[f"sens-pde-{pde_idx}"] = num_sensitive_at_threshold(
                    pde_data["cumulative_variance_fraction"], vt
                )
        elif "per_pde_sensitivity" in data:
            for pde_idx, pde_sens in enumerate(data["per_pde_sensitivity"]):
                total = pde_sens.sum()
                sorted_s, _ = torch.sort(pde_sens, descending=True)
                cum = torch.cumsum(sorted_s, 0) / total
                row[f"sens-pde-{pde_idx}"] = num_sensitive_at_threshold(cum, vt)

        rows.append(row)

    return rows


def main():
    all_rows = []

    # Discover matching .pt files first to get total count
    files_to_process = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        m = FILENAME_PATTERN.match(fname)
        if not m:
            continue
        files_to_process.append((fname, m.group(1), int(m.group(2))))

    total = len(files_to_process)
    print(f"Found {total} ablation .pt files in {RESULTS_DIR}/\n")

    for i, (fname, maxout, nbatch) in enumerate(files_to_process, 1):
        filepath = os.path.join(RESULTS_DIR, fname)

        print(f"[{i}/{total}] Processing: {fname}  (maxout={maxout}, nbatch={nbatch})")
        rows = process_file(filepath, maxout, nbatch)
        all_rows.extend(rows)

    if not all_rows:
        print("No matching .pt files found in", RESULTS_DIR)
        return

    # Determine PDE columns dynamically from the first row
    pde_cols = sorted(
        [k for k in all_rows[0] if k.startswith("sens-pde-") and k != "sens-pde-all"]
    )
    fieldnames = ["nbatch", "maxout", "var_threshold"] + pde_cols + ["sens-pde-all"]

    # Sort rows: by maxout (numeric), then nbatch, then var_threshold
    def sort_key(r):
        try:
            mo = int(r["maxout"])
        except ValueError:
            mo = float("inf")  # "None" → last
        return (mo, r["nbatch"], r["var_threshold"])

    all_rows.sort(key=sort_key)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV written to: {OUTPUT_CSV}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
