"""Generate cross-validation grid-search configs and a Slurm array script.

Run ``python sweep.py`` to read ``grid_search_config.json``. The generated
manifest contains one job for every configured per-field shrinkage-prior
assignment and hyperprior candidate for each study.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import os
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def deep_merge(destination: dict, source: dict) -> dict:
    """Recursively merge source into destination, returning destination."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_merge(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")


def read_theta_count(config: dict) -> int:
    model_dir = Path(config["input_settings"]["model_data_path"])
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    theta_file = model_dir / "thetaVals.txt"
    with theta_file.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    delimiter = config["input_settings"].get("input_delimiter", "\t")
    count = len(header.split(delimiter))
    if count < 1:
        raise ValueError(f"No parameter fields were found in {theta_file}")
    return count


def normalized_candidate(candidate: dict, family: str) -> tuple[str, dict, float | None]:
    if not isinstance(candidate, dict) or not candidate.get("id"):
        raise ValueError(f"Each {family} candidate needs an id and settings")
    settings = copy.deepcopy(candidate.get("settings", {}))
    step_size = candidate.get("step_size")
    if not isinstance(settings, dict):
        raise ValueError(f"Settings for candidate {candidate['id']!r} must be an object")
    if step_size is not None and (not isinstance(step_size, (int, float)) or step_size <= 0):
        raise ValueError(f"step_size for candidate {candidate['id']!r} must be positive")
    return str(candidate["id"]), settings, float(step_size) if step_size is not None else None


def field_candidates(grid: dict) -> list[dict]:
    candidates = []
    supported = ("gp", "lasso", "fused_lasso", "spike_slab")
    for family in supported:
        entries = grid.get(family, [])
        for item in entries:
            candidate_id, settings, step_size = normalized_candidate(item, family)
            candidates.append({
                "family": family,
                "id": candidate_id,
                "settings": settings,
                "step_size": step_size,
            })
    if not candidates:
        raise ValueError("kappa_hyperprior_grid must define candidates for at least one prior family")
    return candidates


def update_field(field: dict, candidate: dict) -> dict:
    updated = copy.deepcopy(field)
    family = candidate["family"]
    updated["shrinkage_prior"] = family
    if candidate["step_size"] is not None:
        updated["step_size"] = candidate["step_size"]
    updated.setdefault(family, {})
    if not isinstance(updated[family], dict):
        raise ValueError(f"The {family} settings for a kappa field must be an object")
    deep_merge(updated[family], candidate["settings"])
    return updated


def study_base_config(base_config: dict, study: dict) -> dict:
    config = copy.deepcopy(base_config)
    deep_merge(config, study.get("overrides", {}))
    for key in ("model_data_path", "obs_data_path"):
        if key not in config.get("input_settings", {}):
            raise ValueError(f"Study config is missing input_settings.{key}")
        config["input_settings"][key] = (
            str(config["input_settings"][key]).rstrip("/\\") + os.sep
        )
    config["cross_validation_settings"]["conduct_cross_validation"] = True
    return config


def main() -> None:
    sweep_path = ROOT / "grid_search_config.json"
    with sweep_path.open("r", encoding="utf-8") as stream:
        sweep = json.load(stream)

    base_path = Path(sweep.get("base_config", "config.json"))
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with base_path.open("r", encoding="utf-8") as stream:
        base_config = json.load(stream)

    raw_constraints = sweep.get("shared_physical_constraints", "from_base_config")
    if raw_constraints == "from_base_config":
        shared_constraints = copy.deepcopy(
            base_config.get("calibration_settings", {}).get("physical_constraints", {})
        )
    elif isinstance(raw_constraints, dict):
        shared_constraints = copy.deepcopy(raw_constraints)
    else:
        raise ValueError("shared_physical_constraints must be an object or 'from_base_config'")
    if not shared_constraints:
        raise ValueError("The shared physical-constraints block is empty")
    constraints_json = json.dumps(shared_constraints, sort_keys=True, separators=(",", ":"))
    constraints_hash = hashlib.sha256(constraints_json.encode("utf-8")).hexdigest()

    candidates = field_candidates(sweep.get("kappa_hyperprior_grid", {}))
    delta_candidates = sweep.get("delta_eta_gp_grid", [])
    if not delta_candidates:
        raise ValueError("delta_eta_gp_grid must define at least one GP hyperprior candidate")
    parsed_delta = [normalized_candidate(item, "gp") for item in delta_candidates]

    output_root = Path(sweep.get("output_root", "results/cv_grid_search"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()
    config_root = ROOT / sweep.get("generated_config_root", "configs/grid_search")
    if not config_root.is_absolute():
        config_root = ROOT / config_root
    config_root = config_root.resolve()
    manifest_path = config_root / "manifest.json"
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    study_configs = sweep.get("studies", {})
    if not study_configs:
        raise ValueError("At least one study must be defined")

    figure_options = sweep.get("figure_options")
    for study_name, study in study_configs.items():
        base = study_base_config(base_config, study)
        if figure_options is not None:
            deep_merge(base.setdefault("output_settings", {}).setdefault("figure_options", {}), figure_options)
        if sweep.get("disable_figures", False):
            options = base.setdefault("output_settings", {}).setdefault("figure_options", {})
            for key, value in list(options.items()):
                if isinstance(value, bool):
                    options[key] = False

        theta_count = read_theta_count(base)
        kappa_fields = base.get("hyperprior_settings", {}).get("kappa_fields", {})
        field_keys = [f"kappa{index + 1}" for index in range(theta_count)]
        missing = [key for key in field_keys if key not in kappa_fields]
        if missing:
            raise ValueError(
                f"{study_name} needs per-field settings in hyperprior_settings.kappa_fields: {missing}"
            )

        combinations = itertools.product(candidates, repeat=theta_count)
        for field_combo in combinations:
            for delta_id, delta_settings, _ in parsed_delta:
                config = copy.deepcopy(base)
                method_parts = []
                candidate_ids = {}
                for index, candidate in enumerate(field_combo):
                    field_key = field_keys[index]
                    config["hyperprior_settings"]["kappa_fields"][field_key] = update_field(
                        config["hyperprior_settings"]["kappa_fields"][field_key], candidate
                    )
                    method_parts.append(f"{field_key}-{candidate['family']}")
                    candidate_ids[field_key] = candidate["id"]

                delta_config = config["hyperprior_settings"].setdefault("delta_eta", {})
                delta_config["shrinkage_prior"] = "gp"
                delta_config.setdefault("gp", {})
                deep_merge(delta_config["gp"], delta_settings)
                candidate_ids["delta_eta"] = delta_id

                method_group = "_".join(method_parts)
                setting_label = "__".join(
                    [f"{key}-{safe_name(value)}" for key, value in candidate_ids.items()]
                )
                run_name = safe_name(f"{setting_label}")
                result_dir = output_root / study_name / method_group / run_name
                config_path = config_root / study_name / f"{run_name}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)

                config["calibration_settings"]["physical_constraints"] = copy.deepcopy(shared_constraints)
                config.setdefault("output_settings", {})["results_path"] = str(
                    result_dir.relative_to(ROOT)
                )
                config["output_settings"]["figures_path"] = str(
                    (result_dir / "figures").relative_to(ROOT)
                )
                config["grid_search_metadata"] = {
                    "study": study_name,
                    "method_group": method_group,
                    "candidate_ids": candidate_ids,
                    "shared_constraints_sha256": constraints_hash,
                }
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                jobs.append({
                    "index": len(jobs),
                    "study": study_name,
                    "method_group": method_group,
                    "candidate_ids": candidate_ids,
                    "config_path": str(config_path.relative_to(ROOT)),
                    "results_path": str(result_dir.relative_to(ROOT)),
                })

    manifest = {
        "schema_version": 1,
        "project_root": "../..",
        "results_root": str(output_root.relative_to(ROOT)),
        "shared_constraints_sha256": constraints_hash,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    template_path = ROOT / "run_array.sbatch"
    template = template_path.read_text(encoding="utf-8")
    concurrency = int(sweep.get("array_concurrency", 16))
    if concurrency < 1:
        raise ValueError("array_concurrency must be a positive integer")
    array_range = f"0-{len(jobs) - 1}%{concurrency}"
    generated_script = template.replace("__ARRAY_RANGE__", array_range)
    generated_script = generated_script.replace(
        "__MANIFEST_PATH__", "configs/grid_search/manifest.json"
    )
    generated_script = generated_script.replace(
        "__LOG_DIR__", str(log_dir.relative_to(ROOT))
    )
    generated_path = config_root / "run_array.generated.sbatch"
    generated_path.write_text(generated_script, encoding="utf-8")

    print(f"Generated {len(jobs)} calibration configs across {len(study_configs)} studies.")
    print(f"Shared physical-constraints SHA-256: {constraints_hash}")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")
    print(
        "Submit the generated array with: "
        f"sbatch {generated_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
