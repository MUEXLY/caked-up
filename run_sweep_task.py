"""Execute one config from a sweep manifest by Slurm array index."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python run_sweep_task.py <manifest.json> <array_index>")
    manifest_path = Path(sys.argv[1]).expanduser().resolve()
    try:
        index = int(sys.argv[2])
    except ValueError as exc:
        raise SystemExit(f"Array index must be an integer, got {sys.argv[2]!r}") from exc

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    jobs = manifest.get("jobs", [])
    if index < 0 or index >= len(jobs):
        raise SystemExit(f"Array index {index} is outside the manifest range 0..{len(jobs)-1}")

    job = jobs[index]
    project_root = Path(manifest["project_root"])
    if not project_root.is_absolute():
        project_root = manifest_path.parent / project_root
    project_root = project_root.resolve()
    config_path = Path(job["config_path"])
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    print(
        f"Starting job {index + 1}/{len(jobs)}: "
        f"study={job['study']} methods={job['method_group']} config={config_path}",
        flush=True,
    )
    subprocess.run(
        [sys.executable, str(project_root / "main.py"), str(config_path)],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
