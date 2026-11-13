"""Rank CV grid-search results by metric within each shrinkage-method group."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def infer_method_group(config: dict) -> str:
    metadata = config.get("grid_search_metadata", {})
    if metadata.get("method_group"):
        return str(metadata["method_group"])
    fields = config.get("hyperprior_settings", {}).get("kappa_fields", {})
    methods = []
    for name in sorted(fields, key=lambda item: int(re.search(r"\d+", item).group())):
        method = fields[name].get("shrinkage_prior", "unknown")
        methods.append(f"{name}-{method}")
    return "_".join(methods) if methods else "unknown"


def extract_run(loss_path: Path, metric: str) -> dict | None:
    losses = load_json(loss_path)
    if not isinstance(losses, dict):
        return None
    run_dir = loss_path.parent
    config = load_json(run_dir / "used_config.json") or load_json(run_dir / "config.json")
    if not isinstance(config, dict):
        return None
    metadata = config.get("grid_search_metadata", {})
    value = losses.get(metric)
    if value is None and metric == "net_nrmse":
        value = losses.get("y_nrmse")
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = None
    return {
        "study": metadata.get("study") or run_dir.parent.parent.name,
        "method_group": infer_method_group(config),
        "rank": None,
        "run_name": run_dir.name,
        "metric": metric,
        "nrmse": value,
        "net_nrmse": losses.get("net_nrmse"),
        "y_nrmse": losses.get("y_nrmse"),
        "theta_nrmse_total": losses.get("theta_nrmse_total"),
        "delta_nrmse": losses.get("delta_nrmse"),
        "holdout_y_nrmse": losses.get("holdout_y_nrmse"),
        "candidate_ids": json.dumps(metadata.get("candidate_ids", {}), sort_keys=True),
        "shared_constraints_sha256": metadata.get("shared_constraints_sha256", ""),
        "results_path": str(run_dir.resolve()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "study", "method_group", "rank", "run_name", "metric", "nrmse",
        "net_nrmse", "y_nrmse", "theta_nrmse_total", "delta_nrmse",
        "holdout_y_nrmse", "candidate_ids", "shared_constraints_sha256", "results_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_study_plots(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV ranking files were still generated")
        return

    by_study: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["nrmse"] is not None and row["rank"] == 1:
            by_study[row["study"]].append(row)
    for study, best_rows in by_study.items():
        best_rows.sort(key=lambda row: row["nrmse"])
        fig_height = max(4.5, 0.35 * len(best_rows))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        names = [row["method_group"].replace("_", "\n", 1) for row in best_rows]
        values = [row["nrmse"] for row in best_rows]
        bars = ax.barh(names, values, color="#4472a8")
        ax.invert_yaxis()
        ax.set_xlabel("net_nrmse (lower is better)")
        ax.set_title(f"{study}: best hyperprior setting by kappa method combination")
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.4g}", va="center")
        fig.tight_layout()
        fig.savefig(output_dir / f"best_methods_{study}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "cv_grid_search"),
        help="Root containing the study/method/run result hierarchy",
    )
    parser.add_argument(
        "--metric",
        default="net_nrmse",
        choices=("net_nrmse", "y_nrmse", "theta_nrmse_total", "delta_nrmse"),
        help="Metric used to rank within each study and method group",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = results_root / "postprocessed"
    loss_paths = sorted(results_root.rglob("cross_validation_losses.json"))
    rows = [row for path in loss_paths if (row := extract_run(path, args.metric)) is not None]
    if not rows:
        raise SystemExit(f"No completed cross-validation results found under {results_root}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], row["method_group"])].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: float("inf") if row["nrmse"] is None else row["nrmse"])
        for rank, row in enumerate(group_rows, start=1):
            row["rank"] = rank

    rows.sort(key=lambda row: (row["study"], row["method_group"], row["rank"]))
    write_csv(output_dir / "ranked_runs.csv", rows)
    best_rows = [row for row in rows if row["rank"] == 1 and row["nrmse"] is not None]
    best_rows.sort(key=lambda row: (row["study"], row["nrmse"]))
    write_csv(output_dir / "best_by_method.csv", best_rows)
    create_study_plots(rows, output_dir)

    print(f"Ranked {len(rows)} completed runs by {args.metric} within study/method groups.")
    print(f"Full ranking: {output_dir / 'ranked_runs.csv'}")
    print(f"Best settings per method group: {output_dir / 'best_by_method.csv'}")
    for row in best_rows:
        print(
            f"{row['study']} | {row['method_group']} | "
            f"{args.metric}={row['nrmse']:.6g} | {row['run_name']}"
        )


if __name__ == "__main__":
    main()
