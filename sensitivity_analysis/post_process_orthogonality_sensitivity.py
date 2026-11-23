"""Summarize orthogonality-sensitivity calibration runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def case_metadata(config: dict) -> dict:
    metadata = config.get("orthogonality_sensitivity_metadata", {})
    orthogonalization = config.get("calibration_settings", {}).get(
        "orthogonalization_settings", {}
    )
    return {
        "study": metadata.get("study", ""),
        "case": metadata.get("case", ""),
        "label": metadata.get("label", metadata.get("case", "")),
        "orthogonalize_kappa": metadata.get(
            "orthogonalize_kappa",
            orthogonalization.get("orthogonalize_kappa", True),
        ),
        "orthogonalize_delta_eta": metadata.get(
            "orthogonalize_delta_eta",
            orthogonalization.get("orthogonalize_delta_eta", True),
        ),
    }


def extract_run(losses_path: Path) -> dict | None:
    losses = read_json(losses_path)
    if not isinstance(losses, dict):
        return None
    run_dir = losses_path.parent
    config = read_json(run_dir / "used_config.json") or read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        return None
    metadata = case_metadata(config)
    return {
        "study": metadata["study"],
        "case": metadata["case"],
        "label": metadata["label"],
        "orthogonalize_kappa": metadata["orthogonalize_kappa"],
        "orthogonalize_delta_eta": metadata["orthogonalize_delta_eta"],
        "net_nrmse": losses.get("net_nrmse"),
        "y_nrmse": losses.get("y_nrmse"),
        "theta_nrmse_total": losses.get("theta_nrmse_total"),
        "delta_nrmse": losses.get("delta_nrmse"),
        "holdout_y_nrmse": losses.get("holdout_y_nrmse"),
        "results_path": str(run_dir.resolve()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "study",
        "case",
        "label",
        "orthogonalize_kappa",
        "orthogonalize_delta_eta",
        "net_nrmse",
        "y_nrmse",
        "theta_nrmse_total",
        "delta_nrmse",
        "holdout_y_nrmse",
        "results_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_plots(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV summaries were still generated")
        return

    preferred_order = [
        "full_orthogonality",
        "no_orthogonality",
        "kappa_only",
        "delta_only",
    ]
    by_study = defaultdict(list)
    for row in rows:
        by_study[row["study"]].append(row)

    for study, study_rows in by_study.items():
        row_map = {row["case"]: row for row in study_rows}
        ordered = [
            row_map[case]
            for case in preferred_order
            if case in row_map
        ]
        ordered.extend(
            row for row in study_rows
            if row["case"] not in preferred_order
        )
        labels = [row["label"] or row["case"] for row in ordered]
        metrics = ("net_nrmse", "y_nrmse", "theta_nrmse_total", "delta_nrmse")
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            squeeze=False,
        )
        for ax, metric in zip(axes.ravel(), metrics):
            values = [row.get(metric) for row in ordered]
            finite = [
                (index, float(value))
                for index, value in enumerate(values)
                if value is not None
            ]
            ax.bar(
                [index for index, _ in finite],
                [value for _, value in finite],
                color=["tab:blue", "tab:orange", "tab:green", "tab:red"][:len(finite)],
            )
            ax.set_xticks(range(len(ordered)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{study}: orthogonality sensitivity")
        fig.tight_layout()
        fig.savefig(output_dir / f"{study}_orthogonality_sensitivity.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "orthogonality_sensitivity"),
        help="Root containing orthogonality-sensitivity result directories",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    rows = [
        row
        for path in sorted(results_root.rglob("cross_validation_losses.json"))
        if (row := extract_run(path)) is not None
    ]
    if not rows:
        raise SystemExit(
            f"No completed cross-validation results found under {results_root}"
        )

    rows.sort(key=lambda row: (
        row["study"],
        row["case"],
    ))
    output_dir = results_root / "postprocessed"
    write_csv(output_dir / "orthogonality_sensitivity.csv", rows)
    create_plots(rows, output_dir)
    print(
        f"Summarized {len(rows)} orthogonality-sensitivity runs."
    )
    print(
        f"CSV: {output_dir / 'orthogonality_sensitivity.csv'}"
    )


if __name__ == "__main__":
    main()
