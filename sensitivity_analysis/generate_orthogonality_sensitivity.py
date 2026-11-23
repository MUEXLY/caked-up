"""Generate orthogonality-constraint calibration configs and a Slurm array."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SENSITIVE_CASES = (
    "full_orthogonality",
    "no_orthogonality",
    "kappa_only",
    "delta_only",
)


def deep_merge(destination: dict, source: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_merge(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")


def main() -> None:
    sweep_path = SCRIPT_DIR / "orthogonality_sensitivity_config.json"
    with sweep_path.open("r", encoding="utf-8") as stream:
        sweep = json.load(stream)

    base_path = Path(sweep.get("base_config", "config.json"))
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with base_path.open("r", encoding="utf-8") as stream:
        base_config = json.load(stream)

    cases = sweep.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must contain at least one orthogonality case")

    seen_case_ids = set()
    normalized_cases = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("id"):
            raise ValueError("Each orthogonality case needs an id")
        case_id = str(case["id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate orthogonality case id: {case_id}")
        seen_case_ids.add(case_id)
        if "orthogonalize_kappa" not in case or "orthogonalize_delta_eta" not in case:
            raise ValueError(
                f"Case {case_id} needs orthogonalize_kappa and "
                "orthogonalize_delta_eta booleans"
            )
        normalized_cases.append({
            "id": case_id,
            "label": str(case.get("label", case_id)),
            "orthogonalize_kappa": bool(case["orthogonalize_kappa"]),
            "orthogonalize_delta_eta": bool(case["orthogonalize_delta_eta"]),
        })

    case_ids = {case["id"] for case in normalized_cases}
    missing_cases = set(SENSITIVE_CASES) - case_ids
    if missing_cases:
        raise ValueError(
            "Orthogonality sensitivity config is missing required cases: "
            + ", ".join(sorted(missing_cases))
        )

    studies = sweep.get("studies")
    if not isinstance(studies, dict) or not studies:
        raise ValueError("studies must contain at least one study")

    output_root = Path(sweep.get("output_root", "results/orthogonality_sensitivity"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()

    config_root = Path(
        sweep.get("generated_config_root", "configs/orthogonality_sensitivity")
    )
    if not config_root.is_absolute():
        config_root = ROOT / config_root
    config_root = config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    repeats = int(sweep.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")
    base_seed = int(sweep.get("base_seed", 42000))

    jobs = []
    for study_name, study in studies.items():
        if not isinstance(study, dict):
            raise ValueError(f"Study {study_name} must be an object")
        base = copy.deepcopy(base_config)
        deep_merge(base, study.get("overrides", {}))
        input_settings = base.setdefault("input_settings", {})
        for key in ("model_data_path", "obs_data_path"):
            if key not in input_settings:
                raise ValueError(f"{study_name} is missing input_settings.{key}")
            input_settings[key] = str(input_settings[key]).rstrip("/\\") + "/"
        base.setdefault("cross_validation_settings", {})["conduct_cross_validation"] = True

        if sweep.get("disable_figures", False):
            figure_options = base.setdefault("output_settings", {}).setdefault(
                "figure_options", {}
            )
            for key, value in list(figure_options.items()):
                if isinstance(value, bool):
                    figure_options[key] = False

        for case in normalized_cases:
            for repeat in range(1, repeats + 1):
                config = copy.deepcopy(base)
                seed = base_seed + len(jobs)
                config["random_seed"] = seed
                config["calibration_settings"]["random_seed"] = seed
                orthogonalization = config.setdefault(
                    "calibration_settings", {}
                ).setdefault("orthogonalization_settings", {})
                orthogonalization["orthogonalize_kappa"] = case["orthogonalize_kappa"]
                orthogonalization["orthogonalize_delta_eta"] = case["orthogonalize_delta_eta"]

                run_name = safe_name(case["id"])
                result_dir = output_root / study_name / f"{run_name}_repeat_{repeat:02d}"
                config_path = config_root / study_name / f"{run_name}_repeat_{repeat:02d}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)

                config.setdefault("output_settings", {})["results_path"] = str(
                    result_dir.relative_to(ROOT)
                )
                config["output_settings"]["figures_path"] = str(
                    (result_dir / "figures").relative_to(ROOT)
                )
                config["orthogonality_sensitivity_metadata"] = {
                    "study": study_name,
                    "case": case["id"],
                    "label": case["label"],
                    "repeat": repeat,
                    "seed": seed,
                    "orthogonalize_kappa": case["orthogonalize_kappa"],
                    "orthogonalize_delta_eta": case["orthogonalize_delta_eta"],
                }
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                jobs.append({
                    "index": len(jobs),
                    "study": study_name,
                    "case": case["id"],
                    "label": case["label"],
                    "repeat": repeat,
                    "seed": seed,
                    "orthogonalize_kappa": case["orthogonalize_kappa"],
                    "orthogonalize_delta_eta": case["orthogonalize_delta_eta"],
                    "config_path": str(config_path.relative_to(ROOT)),
                    "results_path": str(result_dir.relative_to(ROOT)),
                })

    manifest_path = config_root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "project_root": "../..",
        "results_root": str(output_root.relative_to(ROOT)),
        "job_count": len(jobs),
        "repeats": repeats,
        "base_seed": base_seed,
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    template_path = SCRIPT_DIR / "run_orthogonality_sensitivity.sbatch"
    template = template_path.read_text(encoding="utf-8")
    concurrency = int(sweep.get("array_concurrency", 4))
    if concurrency < 1:
        raise ValueError("array_concurrency must be a positive integer")
    generated = template.replace(
        "__ARRAY_RANGE__", f"0-{len(jobs) - 1}%{concurrency}"
    )
    generated = generated.replace(
        "__MANIFEST_PATH__", "configs/orthogonality_sensitivity/manifest.json"
    )
    generated = generated.replace(
        "__LOG_DIR__", str((output_root / "logs").relative_to(ROOT))
    )
    generated_path = config_root / "run_orthogonality_sensitivity.generated.sbatch"
    generated_path.write_text(generated, encoding="utf-8")

    print(
        f"Generated {len(jobs)} orthogonality configs "
        f"for {len(studies)} studies."
    )
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")
    print(
        "Submit the generated array with: "
        f"sbatch {generated_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
