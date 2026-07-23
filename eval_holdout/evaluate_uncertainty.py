"""Holdout evaluation driver for CAKED-UP.

This script repeatedly withholds a configurable fraction of the observation
data, rewrites a temporary calibration config so `main.py` uses that subset,
and then collects the calibration framework's holdout NRMSE output.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _resolve_path(base_dir: Path, candidate: str) -> Path:
	path = Path(candidate)
	if path.is_absolute():
		return path
	return (base_dir / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=4)


def _read_text_table(path: Path) -> tuple[str, list[str]]:
	with path.open("r", encoding="utf-8") as handle:
		lines = [line.rstrip("\n") for line in handle if line.strip()]

	if not lines:
		raise ValueError(f"File is empty: {path}")

	return lines[0], lines[1:]


def _write_subset_table(
	source_path: Path,
	target_path: Path,
	selected_indices: np.ndarray,
) -> None:
	header, rows = _read_text_table(source_path)
	target_path.parent.mkdir(parents=True, exist_ok=True)

	with target_path.open("w", encoding="utf-8", newline="\n") as handle:
		handle.write(f"{header}\n")
		for row_index in selected_indices:
			handle.write(f"{rows[int(row_index)]}\n")


def _prepare_holdout_split(
	source_obs_dir: Path,
	split_obs_dir: Path,
	holdout_fraction: float,
	seed: int,
) -> dict[str, Any]:
	source_app_path = source_obs_dir / "appDomain.txt"
	source_y_path = source_obs_dir / "observationData.txt"

	if not source_app_path.exists():
		raise FileNotFoundError(f"Missing observation input file: {source_app_path}")
	if not source_y_path.exists():
		raise FileNotFoundError(f"Missing observation input file: {source_y_path}")

	_, app_rows = _read_text_table(source_app_path)
	_, y_rows = _read_text_table(source_y_path)

	if len(app_rows) != len(y_rows):
		raise ValueError(
			"Observation appDomain and observationData files must contain the same number of rows."
		)

	n_total = len(app_rows)
	if n_total < 2:
		raise ValueError("Need at least two observations to form a holdout split.")

	n_holdout = int(round(n_total * holdout_fraction))
	n_holdout = max(1, n_holdout)
	n_holdout = min(n_holdout, n_total - 1)

	rng = np.random.default_rng(seed)
	holdout_indices = np.sort(rng.choice(n_total, size=n_holdout, replace=False))
	keep_mask = np.ones(n_total, dtype=bool)
	keep_mask[holdout_indices] = False
	keep_indices = np.flatnonzero(keep_mask)

	split_obs_dir.mkdir(parents=True, exist_ok=True)

	_write_subset_table(source_app_path, split_obs_dir / "appDomain.txt", keep_indices)
	_write_subset_table(source_y_path, split_obs_dir / "observationData.txt", keep_indices)
	_write_subset_table(source_app_path, split_obs_dir / "appDomain_holdout.txt", holdout_indices)
	_write_subset_table(source_y_path, split_obs_dir / "observationData_holdout.txt", holdout_indices)

	return {
		"n_total": n_total,
		"n_holdout": int(n_holdout),
		"n_keep": int(n_total - n_holdout),
		"holdout_indices": holdout_indices.tolist(),
		"keep_indices": keep_indices.tolist(),
	}


def _build_run_config(
	base_config: dict[str, Any],
	split_obs_dir: Path,
	results_dir: Path,
	split_obs_dir_name: str,
) -> dict[str, Any]:
	run_config = copy.deepcopy(base_config)

	run_config.setdefault("input_settings", {})
	run_config.setdefault("cross_validation_settings", {})
	run_config.setdefault("output_settings", {})

	run_config["input_settings"]["obs_data_path"] = f"{split_obs_dir.as_posix()}/"
	run_config["cross_validation_settings"]["holdout_data"] = True
	run_config["cross_validation_settings"]["holdout_data_paths"] = {
		"appDomain_holdout": str(split_obs_dir / "appDomain_holdout.txt"),
		"observationData_holdout": str(split_obs_dir / "observationData_holdout.txt"),
	}
	run_config["output_settings"]["results_path"] = str(results_dir)
	run_config["output_settings"]["figures_path"] = str(results_dir / "figures")
	run_config["output_settings"]["evaluation_label"] = split_obs_dir_name

	return run_config


def _run_main(repo_root: Path, main_script: Path, config_path: Path, log_path: Path) -> None:
	command = [sys.executable, str(main_script), str(config_path)]
	completed = subprocess.run(
		command,
		cwd=str(repo_root),
		capture_output=True,
		text=True,
	)

	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as handle:
		handle.write(f"Command: {' '.join(command)}\n\n")
		handle.write("STDOUT:\n")
		handle.write(completed.stdout)
		handle.write("\n\nSTDERR:\n")
		handle.write(completed.stderr)

	if completed.returncode != 0:
		raise RuntimeError(
			f"Calibration run failed for {config_path.name}. See {log_path} for details."
		)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
	grouped: dict[str, list[float]] = {}
	for record in records:
		grouped.setdefault(record["holdout_fraction_label"], []).append(
			float(record["holdout_y_nrmse"])
		)

	summary: dict[str, Any] = {}
	for label, values in grouped.items():
		array = np.asarray(values, dtype=float)
		summary[label] = {
			"mean_holdout_y_nrmse": float(np.mean(array)),
			"std_holdout_y_nrmse": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
			"min_holdout_y_nrmse": float(np.min(array)),
			"max_holdout_y_nrmse": float(np.max(array)),
			"repetitions": int(len(array)),
		}

	return summary


def _plot_summary(records: list[dict[str, Any]], plot_path: Path) -> None:
	grouped: dict[float, list[float]] = {}
	for record in records:
		grouped.setdefault(float(record["holdout_fraction"]), []).append(
			float(record["holdout_y_nrmse"])
		)

	fractions = sorted(grouped.keys())
	means = [float(np.mean(grouped[fraction])) for fraction in fractions]
	errors = [float(np.std(grouped[fraction], ddof=1)) if len(grouped[fraction]) > 1 else 0.0 for fraction in fractions]

	plt.figure(figsize=(8, 5))
	plt.errorbar(
		[fraction * 100.0 for fraction in fractions],
		means,
		yerr=errors,
		fmt="o-",
		capsize=4,
		linewidth=2,
	)
	plt.xlabel("Data withheld from calibration (%)")
	plt.ylabel("Holdout NRMSE")
	plt.title("Holdout error vs. withheld data")
	plt.grid(True, alpha=0.25)
	plt.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(plot_path, dpi=200)
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Run holdout uncertainty evaluation.")
	parser.add_argument(
		"--config",
		default=None,
		help="Path to the holdout evaluation settings JSON.",
	)
	args = parser.parse_args()

	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent
	config_path = Path(args.config).resolve() if args.config else script_dir / "config_holdout_eval.json"

	evaluation_config = _load_json(config_path)
	base_config_path = _resolve_path(config_path.parent, evaluation_config["base_config_path"])
	base_config = _load_json(base_config_path)

	source_obs_dir = _resolve_path(base_config_path.parent, base_config["input_settings"]["obs_data_path"])
	main_script = repo_root / "main.py"
	if not main_script.exists():
		raise FileNotFoundError(f"Could not find main.py at {main_script}")

	evaluation_root = _resolve_path(config_path.parent, evaluation_config.get("evaluation_root", "./holdout_evaluation"))
	configs_root = evaluation_root / "configs"
	data_root = evaluation_root / "data"
	results_root = evaluation_root / "results"
	logs_root = evaluation_root / "logs"
	for directory in (configs_root, data_root, results_root, logs_root):
		directory.mkdir(parents=True, exist_ok=True)

	holdout_percentages = evaluation_config["holdout_percentages"]
	holdout_repetitions = int(evaluation_config["holdout_repetitions"])
	base_seed = int(evaluation_config.get("random_seed", 42))

	records: list[dict[str, Any]] = []

	for pct_index, holdout_fraction in enumerate(holdout_percentages):
		fraction_label = f"{int(round(float(holdout_fraction) * 100)):02d}pct"
		for repetition in range(holdout_repetitions):
			run_name = f"holdout_{fraction_label}_rep{repetition + 1:02d}"
			run_data_dir = data_root / run_name / "observationData"
			run_results_dir = results_root / run_name
			run_config_path = configs_root / f"{run_name}.json"
			run_log_path = logs_root / f"{run_name}.log"

			split_metadata = _prepare_holdout_split(
				source_obs_dir=source_obs_dir,
				split_obs_dir=run_data_dir,
				holdout_fraction=float(holdout_fraction),
				seed=base_seed + pct_index * 1000 + repetition,
			)

			run_config = _build_run_config(
				base_config=base_config,
				split_obs_dir=run_data_dir,
				results_dir=run_results_dir,
				split_obs_dir_name=run_name,
			)

			_write_json(run_config_path, run_config)
			_run_main(repo_root=repo_root, main_script=main_script, config_path=run_config_path, log_path=run_log_path)

			loss_path = run_results_dir / "cross_validation_losses.json"
			if not loss_path.exists():
				raise FileNotFoundError(f"Missing calibration losses file: {loss_path}")

			loss_data = _load_json(loss_path)
			holdout_y_nrmse = loss_data.get("holdout_y_nrmse")
			if holdout_y_nrmse is None:
				raise ValueError(
					f"Calibration run {run_name} did not produce holdout_y_nrmse."
				)

			records.append(
				{
					"run_name": run_name,
					"holdout_fraction": float(holdout_fraction),
					"holdout_fraction_label": f"{float(holdout_fraction):.3f}",
					"repetition": repetition + 1,
					"holdout_y_nrmse": float(holdout_y_nrmse),
					"results_path": str(run_results_dir),
					"config_path": str(run_config_path),
					"log_path": str(run_log_path),
					"n_total": split_metadata["n_total"],
					"n_holdout": split_metadata["n_holdout"],
					"n_keep": split_metadata["n_keep"],
				}
			)

			print(
				f"{run_name}: holdout_y_nrmse={float(holdout_y_nrmse):.6f} "
				f"(n_keep={split_metadata['n_keep']}, n_holdout={split_metadata['n_holdout']})"
			)

	summary = {
		"base_config_path": str(base_config_path),
		"source_observation_dir": str(source_obs_dir),
		"evaluation_root": str(evaluation_root),
		"holdout_percentages": holdout_percentages,
		"holdout_repetitions": holdout_repetitions,
		"random_seed": base_seed,
		"records": records,
		"summary_by_fraction": _summarize(records),
	}

	summary_path = _resolve_path(config_path.parent, evaluation_config.get("summary_file", "./holdout_evaluation/holdout_summary.json"))
	_write_json(summary_path, summary)

	csv_path = summary_path.with_suffix(".csv")
	with csv_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"run_name",
				"holdout_fraction",
				"repetition",
				"holdout_y_nrmse",
				"n_total",
				"n_holdout",
				"n_keep",
				"results_path",
				"config_path",
				"log_path",
			],
		)
		writer.writeheader()
		for record in records:
			writer.writerow({key: record[key] for key in writer.fieldnames})

	plot_path = summary_path.with_name("holdout_nrmse_vs_withheld_data.png")
	_plot_summary(records, plot_path)

	print(f"Saved summary to {summary_path}")
	print(f"Saved CSV summary to {csv_path}")
	print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
	main()
