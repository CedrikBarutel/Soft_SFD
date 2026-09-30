#!/usr/bin/env python3
"""Run staged LAMMPS simulations with reproducible manifests."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List

from simulation_config import DEFAULTS, SimulationDefaults, derived_parameters


STAGE_PRESETS = {
    "stage0": {
        "mode": "diffusion",
        "n_values": [1],
        "width_factors": [3.0],
        "runs": 1,
        "runtime_factor": 50.0,
        "output_root": "results_stage0",
    },
    "stage0_force": {
        "mode": "force",
        "n_values": [1],
        "width_factors": [3.0],
        "runs": 1,
        "runtime_factor": 50.0,
        "output_root": "results_stage0_force",
    },
    "stage1": {
        "mode": "diffusion",
        "n_values": [50],
        "width_factors": [0.5],
        "runs": 3,
        "runtime_factor": 2000.0,
        "output_root": "results_stage1_sfd",
    },
}


def parse_number_list(value: str, cast: type) -> List[Any]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return [cast(item) for item in items]


def parse_int_list(value: str) -> List[int]:
    return parse_number_list(value, int)


def parse_float_list(value: str) -> List[float]:
    return parse_number_list(value, float)


def stage_defaults(stage: str) -> Dict[str, Any]:
    if stage == "custom":
        return {
            "mode": "diffusion",
            "n_values": [1],
            "width_factors": [3.0],
            "runs": 1,
            "runtime_factor": DEFAULTS.runtime_factor,
            "output_root": "results_custom",
        }
    return dict(STAGE_PRESETS[stage])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["custom", *STAGE_PRESETS.keys()],
        default="stage0",
        help="Stage preset to run. Explicit CLI values override the preset.",
    )
    parser.add_argument("--mode", choices=["diffusion", "force"])
    parser.add_argument("--N-values", type=parse_int_list, dest="n_values")
    parser.add_argument("--width-factors", type=parse_float_list)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output-root")
    parser.add_argument("--lammps-exec", default=None)
    parser.add_argument("--runtime-factor", type=float)
    parser.add_argument("--target-frames", type=int, default=DEFAULTS.target_frames)
    parser.add_argument("--length", type=float, default=DEFAULTS.length)
    parser.add_argument("--sigma", type=float, default=DEFAULTS.sigma)
    parser.add_argument("--diffusion-coef", type=float, default=DEFAULTS.diffusion_coef)
    parser.add_argument("--temperature", type=float, default=DEFAULTS.temperature)
    parser.add_argument("--timestep", type=float, default=DEFAULTS.timestep)
    parser.add_argument("--walk-velocity", type=float, default=DEFAULTS.walk_velocity)
    parser.add_argument("--seed-base", type=int, default=1729)
    parser.add_argument("--write-movie", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def merged_args(args: argparse.Namespace) -> Dict[str, Any]:
    preset = stage_defaults(args.stage)
    for key in ("mode", "n_values", "width_factors", "runs", "runtime_factor", "output_root"):
        value = getattr(args, key)
        if value is not None:
            preset[key] = value
    return preset


def template_for(mode: str, script_dir: Path) -> Path:
    name = {
        "diffusion": "LAMMPS_channel_diffusion.in",
        "force": "LAMMPS_channel_force.in",
    }[mode]
    return script_dir / name


def default_lammps_exec(script_dir: Path) -> str:
    candidates = [
        script_dir / "lmp",
        script_dir.parent / "lmp",
        script_dir.parent / "AI_Assisted" / "lmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "./lmp"


def format_width(width_factor: float) -> str:
    return ("%g" % width_factor).replace(".", "p")


def make_jobs(args: argparse.Namespace, settings: Dict[str, Any], script_dir: Path) -> List[Dict[str, Any]]:
    output_root = Path(settings["output_root"])
    results_dir = output_root / "trajectories"
    logs_dir = output_root / "logs"
    template = template_for(settings["mode"], script_dir)
    lammps_exec = args.lammps_exec or default_lammps_exec(script_dir)

    defaults = replace(
        DEFAULTS,
        length=args.length,
        sigma=args.sigma,
        diffusion_coef=args.diffusion_coef,
        temperature=args.temperature,
        timestep=args.timestep,
        walk_velocity=args.walk_velocity,
    )
    rng = Random(args.seed_base)
    jobs: List[Dict[str, Any]] = []

    for n_particles in settings["n_values"]:
        for width_factor in settings["width_factors"]:
            for run_id in range(settings["runs"]):
                params = derived_parameters(
                    n_particles=n_particles,
                    width_factor=width_factor,
                    mode=settings["mode"],
                    defaults=defaults,
                    runtime_factor=settings["runtime_factor"],
                    target_frames=args.target_frames,
                )
                seed = rng.randint(100000, 999999999)
                stem = (
                    f"sim_mode-{settings['mode']}_"
                    f"N{n_particles}_W{format_width(width_factor)}_run{run_id}"
                )
                dump_file = results_dir / f"{stem}.dump"
                movie_file = results_dir / f"{stem}.mp4"
                lammps_log = logs_dir / f"{stem}.lammps.log"
                stdout_file = logs_dir / f"{stem}.stdout.log"
                stderr_file = logs_dir / f"{stem}.stderr.log"
                moviefreq = max(params["dumpfreq"], params["dumpfreq"] * 10)

                command = [
                    lammps_exec,
                    "-in",
                    str(template),
                    "-var",
                    "N",
                    str(n_particles),
                    "-var",
                    "seed",
                    str(seed),
                    "-var",
                    "length",
                    str(params["length"]),
                    "-var",
                    "sigma",
                    str(params["sigma"]),
                    "-var",
                    "diffusion_coef",
                    str(params["diffusion_coef"]),
                    "-var",
                    "temp",
                    str(params["temperature"]),
                    "-var",
                    "timestep",
                    str(params["timestep"]),
                    "-var",
                    "runsteps",
                    str(params["runsteps"]),
                    "-var",
                    "dumpfreq",
                    str(params["dumpfreq"]),
                    "-var",
                    "dumpfile_1",
                    str(dump_file),
                    "-var",
                    "moviefile_1",
                    str(movie_file),
                    "-var",
                    "logfile_name",
                    str(lammps_log),
                    "-var",
                    "width_factor",
                    str(width_factor),
                    "-var",
                    "walk_velocity",
                    str(params["walk_velocity"]),
                    "-var",
                    "write_movie",
                    "1" if args.write_movie else "0",
                    "-var",
                    "moviefreq",
                    str(moviefreq),
                    "-log",
                    "none",
                ]

                jobs.append(
                    {
                        **params,
                        "stage": args.stage,
                        "run_id": run_id,
                        "seed": seed,
                        "output_root": str(output_root),
                        "dump_file": str(dump_file),
                        "movie_file": str(movie_file) if args.write_movie else "",
                        "lammps_log": str(lammps_log),
                        "stdout_file": str(stdout_file),
                        "stderr_file": str(stderr_file),
                        "template": str(template),
                        "command": command,
                        "status": "pending",
                        "returncode": "",
                        "elapsed_seconds": "",
                    }
                )
    return jobs


def write_manifest(output_root: Path, jobs: Iterable[Dict[str, Any]]) -> None:
    rows = list(jobs)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "manifest.csv"
    json_path = output_root / "manifest.json"

    manifest_keys = [
        "stage",
        "mode",
        "N",
        "width_factor",
        "run_id",
        "seed",
        "length",
        "sigma",
        "diffusion_coef",
        "temperature",
        "gamma",
        "timestep",
        "runtime_factor",
        "target_frames",
        "run_time",
        "t_coll",
        "t_sigma",
        "runsteps",
        "dumpfreq",
        "expected_v",
        "dump_file",
        "movie_file",
        "lammps_log",
        "stdout_file",
        "stderr_file",
        "template",
        "status",
        "returncode",
        "elapsed_seconds",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in manifest_keys})

    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    for key in ("dump_file", "lammps_log", "stdout_file", "stderr_file"):
        Path(job[key]).parent.mkdir(parents=True, exist_ok=True)

    with Path(job["stdout_file"]).open("w") as stdout_handle, Path(job["stderr_file"]).open("w") as stderr_handle:
        completed = subprocess.run(
            job["command"],
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )

    job["returncode"] = completed.returncode
    job["elapsed_seconds"] = round(time.time() - start, 3)
    job["status"] = "success" if completed.returncode == 0 and Path(job["dump_file"]).exists() else "failed"
    return job


def print_dry_run(jobs: List[Dict[str, Any]]) -> None:
    print(f"Prepared {len(jobs)} simulation(s).")
    for job in jobs:
        print(" ".join(job["command"]))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = merged_args(args)

    if settings["runs"] < 1:
        parser.error("--runs must be >= 1")
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")

    script_dir = Path(__file__).resolve().parent
    output_root = Path(settings["output_root"])
    jobs = make_jobs(args, settings, script_dir)
    write_manifest(output_root, jobs)

    if args.dry_run:
        print_dry_run(jobs)
        print(f"Wrote dry-run manifest to {output_root / 'manifest.csv'}")
        return 0

    print(f"Running {len(jobs)} simulation(s) with parallel={args.parallel}")
    completed_jobs: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            job = future.result()
            completed_jobs.append(job)
            print(
                f"{job['status']:7s} N={job['N']} W={job['width_factor']} "
                f"run={job['run_id']} elapsed={job['elapsed_seconds']}s"
            )

    by_key = {(job["N"], job["width_factor"], job["run_id"]): job for job in completed_jobs}
    for index, job in enumerate(jobs):
        jobs[index] = by_key[(job["N"], job["width_factor"], job["run_id"])]
    write_manifest(output_root, jobs)

    failures = [job for job in jobs if job["status"] != "success"]
    if failures:
        print(f"{len(failures)} simulation(s) failed. See per-run stderr logs.", file=sys.stderr)
        return 1

    print(f"All simulations completed. Manifest: {output_root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
