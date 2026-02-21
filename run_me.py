#!/usr/bin/env python3
"""
train_threshold_model.py

Portable threshold “model” for timesheet anomalies (no custom classes in pickle).

For each (Employee, Project) group we compute:
  - mean_hours
  - mode_hours (most frequent exact value; if tie, smallest)

Then define HIGH vs LOW and set a rule:

HIGH group:
  - if mean == mode (within tolerance): flag anything BELOW mode
  - else: flag anything BELOW min(mean, mode) rounded DOWN to 0.5

LOW group:
  - if mean == mode (within tolerance): flag anything ABOVE mode
  - else: flag anything ABOVE max(mean, mode) rounded UP to 0.5

We save:
  - ./models/<model_name>.pkl  (portable: dict of primitive types only)
  - an Excel summary showing behaviour across a test grid (0..12 in 0.5 steps)

Dates:
  - Timesheet Date is NOT used as a feature. A 2026 entry is still scored normally.
"""

import argparse
import os
import pickle
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# ----------------------------
# 0.5 increment rounding
# ----------------------------
def round_down_half(x: float) -> float:
    return float(np.floor(x * 2.0) / 2.0)


def round_up_half(x: float) -> float:
    return float(np.ceil(x * 2.0) / 2.0)


def is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ----------------------------
# Robust mode (ties handled)
# ----------------------------
def robust_mode(values: np.ndarray) -> Optional[float]:
    """
    Returns the most frequent value.
    If tie, returns the smallest value among tied modes.
    """
    if values.size == 0:
        return None
    s = pd.Series(values)
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return None
    max_count = vc.iloc[0]
    modes = vc[vc == max_count].index.to_list()
    return float(min(modes))


# ----------------------------
# Scoring helper (portable)
# ----------------------------
def flag_with_rule(rule: dict, hours: float) -> bool:
    direction = rule["direction"]  # "below" or "above"
    threshold = float(rule["threshold"])
    if direction == "below":
        return float(hours) < threshold
    if direction == "above":
        return float(hours) > threshold
    raise ValueError(f"Unknown direction: {direction}")


# ----------------------------
# Core training logic (returns dict, not class)
# ----------------------------
def derive_group_rule(
    employee: str,
    project: str,
    hours: np.ndarray,
    high_cutoff: float,
    mean_mode_tol: float,
) -> Optional[dict]:
    hours = hours[~np.isnan(hours)]
    n = int(hours.size)
    if n == 0:
        return None

    mean_h = float(np.mean(hours))
    mode_h = robust_mode(hours)
    if mode_h is None:
        return None

    # Decide HIGH vs LOW: HIGH if BOTH mean and mode >= high_cutoff
    is_high = (mean_h >= high_cutoff) and (mode_h >= high_cutoff)
    group_level = "HIGH" if is_high else "LOW"

    same = is_close(mean_h, mode_h, tol=mean_mode_tol)

    if is_high:
        if same:
            threshold = round_down_half(mode_h)
            rule_name = "FLAG_BELOW_MODE"
        else:
            threshold = round_down_half(min(mean_h, mode_h))
            rule_name = "FLAG_BELOW_MIN(mean,mode)_DOWN_0.5"
        direction = "below"
    else:
        if same:
            threshold = round_up_half(mode_h)
            rule_name = "FLAG_ABOVE_MODE"
        else:
            threshold = round_up_half(max(mean_h, mode_h))
            rule_name = "FLAG_ABOVE_MAX(mean,mode)_UP_0.5"
        direction = "above"

    return {
        "employee": employee,
        "project": project,
        "n_rows": n,
        "mean_hours": mean_h,
        "mode_hours": float(mode_h),
        "group_level": group_level,   # "HIGH" or "LOW"
        "rule": rule_name,
        "threshold": float(threshold),
        "direction": direction,       # "below" or "above"
    }


def make_test_grid(min_h: float = 0.0, max_h: float = 12.0, step: float = 0.5) -> np.ndarray:
    n_steps = int(round((max_h - min_h) / step)) + 1
    return np.round(min_h + np.arange(n_steps) * step, 10)


def find_first_flagged(rule: dict, grid: np.ndarray) -> Optional[float]:
    flags = np.array([flag_with_rule(rule, float(x)) for x in grid], dtype=bool)
    idx = np.where(flags)[0]
    if idx.size == 0:
        return None
    return float(grid[idx[0]])


# ----------------------------
# IO helpers
# ----------------------------
def ensure_models_dir(script_dir: str) -> str:
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", default="scores_log.xlsx", help="Path to cleaned aggregated Excel")
    ap.add_argument("--sheet", default="Training Data", help="Sheet name, e.g. 'Training Data'")
    ap.add_argument("--employee_col", default="Name", help="Employee column name")
    ap.add_argument("--project_col", default="Project Name", help="Project column name")
    ap.add_argument("--hours_col", default="Time in hours", help="Hours column name")
    ap.add_argument("--min_samples", type=int, default=1, help="Minimum rows per (employee,project) to build rule")
    ap.add_argument("--high_cutoff", type=float, default=4.0, help="Threshold for deciding HIGH vs LOW")
    ap.add_argument("--mean_mode_tol", type=float, default=1e-9, help="Tolerance to treat mean==mode")
    ap.add_argument("--grid_min", type=float, default=0.0, help="Test grid min hours")
    ap.add_argument("--grid_max", type=float, default=12.0, help="Test grid max hours")
    ap.add_argument("--grid_step", type=float, default=0.5, help="Test grid step (0.5 for half-hours)")
    ap.add_argument("--out_summary_xlsx", default="model_summary.xlsx", help="Output summary Excel filename")
    ap.add_argument("--model_name", default="halfthreshold_model.pkl", help="Pickle filename saved under ./models/")
    args = ap.parse_args()

    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet)

    required = [args.employee_col, args.project_col, args.hours_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in sheet '{args.sheet}': {missing}\nFound: {list(df.columns)}"
        )

    # Clean
    df = df.copy()
    df[args.employee_col] = df[args.employee_col].astype(str).str.strip()
    df[args.project_col] = df[args.project_col].astype(str).str.strip()
    df[args.hours_col] = pd.to_numeric(df[args.hours_col], errors="coerce")

    df = df.dropna(subset=[args.employee_col, args.project_col, args.hours_col])
    df = df[(df[args.employee_col] != "") & (df[args.project_col] != "")]

    print(f"Training rows (after cleaning): {len(df)}")

    # Train rules per (employee, project)
    rules: Dict[Tuple[str, str], dict] = {}
    skipped: List[dict] = []

    for (emp, proj), g in df.groupby([args.employee_col, args.project_col], dropna=False):
        hours = g[args.hours_col].to_numpy(dtype=float)
        if len(hours) < args.min_samples:
            skipped.append({
                "employee": emp,
                "project": proj,
                "n_rows": int(len(hours)),
                "status": f"SKIPPED (< min_samples={args.min_samples})"
            })
            continue

        rule = derive_group_rule(
            employee=str(emp),
            project=str(proj),
            hours=hours,
            high_cutoff=float(args.high_cutoff),
            mean_mode_tol=float(args.mean_mode_tol),
        )
        if rule is None:
            skipped.append({
                "employee": emp,
                "project": proj,
                "n_rows": int(len(hours)),
                "status": "SKIPPED (could not compute mean/mode)"
            })
            continue

        rules[(str(emp), str(proj))] = rule

    # Save model (portable pickle) in ./models next to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = ensure_models_dir(script_dir)
    model_path = os.path.join(models_dir, args.model_name)

    # IMPORTANT: pickle cannot store tuple keys if you later want JSON-like portability.
    # We'll store rules as a LIST of dicts + an index dict for quick lookup.
    rules_list = list(rules.values())
    index = {f"{r['employee']}||{r['project']}": i for i, r in enumerate(rules_list)}

    model_payload = {
        "type": "threshold_rules_v2_portable",
        "employee_col": args.employee_col,
        "project_col": args.project_col,
        "hours_col": args.hours_col,
        "high_cutoff": float(args.high_cutoff),
        "mean_mode_tol": float(args.mean_mode_tol),
        "min_samples": int(args.min_samples),
        "rules": rules_list,  # list of dicts (portable)
        "index": index,       # str key -> position in rules list
    }
    save_pickle(model_payload, model_path)
    print(f"Saved model pickle: {model_path}")

    # Build comprehensive summary
    grid = make_test_grid(args.grid_min, args.grid_max, args.grid_step)

    rows = []
    for rule in rules_list:
        emp = rule["employee"]
        proj = rule["project"]

        g = df[(df[args.employee_col] == emp) & (df[args.project_col] == proj)]
        hours = g[args.hours_col].to_numpy(dtype=float)

        p05 = float(np.percentile(hours, 5)) if hours.size else np.nan
        p95 = float(np.percentile(hours, 95)) if hours.size else np.nan
        min_h = float(np.min(hours)) if hours.size else np.nan
        max_h = float(np.max(hours)) if hours.size else np.nan

        first_flag = find_first_flagged(rule, grid)

        flag_map = {f"flag_at_{x:g}h": int(flag_with_rule(rule, float(x))) for x in grid}

        base = {
            "employee": emp,
            "project": proj,
            "n_rows": int(rule["n_rows"]),
            "min_train_hours": min_h,
            "max_train_hours": max_h,
            "p05_train_hours": p05,
            "p95_train_hours": p95,
            "mean_hours": float(rule["mean_hours"]),
            "mode_hours": float(rule["mode_hours"]),
            "group_level": rule["group_level"],
            "rule": rule["rule"],
            "direction": rule["direction"],
            "threshold_0.5": float(rule["threshold"]),
            "first_flagged_on_grid": first_flag,
        }
        base.update(flag_map)
        rows.append(base)

    summary_df = pd.DataFrame(rows).sort_values(["employee", "project"])
    skipped_df = pd.DataFrame(skipped).sort_values(["employee", "project"]) if skipped else pd.DataFrame(
        columns=["employee", "project", "n_rows", "status"]
    )

    out_path = os.path.join(os.getcwd(), args.out_summary_xlsx)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, index=False, sheet_name="Rules Summary")
        skipped_df.to_excel(xw, index=False, sheet_name="Skipped Groups")

        compact_cols = [
            "employee", "project", "n_rows",
            "min_train_hours", "max_train_hours", "mean_hours", "mode_hours",
            "group_level", "rule", "direction", "threshold_0.5", "first_flagged_on_grid"
        ]
        summary_df[compact_cols].to_excel(xw, index=False, sheet_name="Compact View")

    print(f"Saved summary Excel: {out_path}")

    print("\nSample (Compact View):")
    if not summary_df.empty:
        print(summary_df[[
            "employee", "project", "n_rows",
            "mean_hours", "mode_hours",
            "group_level", "direction", "threshold_0.5", "rule"
        ]].head(15).to_string(index=False))
    else:
        print("(no rules built)")


if __name__ == "__main__":
    main()
