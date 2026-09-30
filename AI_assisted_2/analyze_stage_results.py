#!/usr/bin/env python3
"""Create trajectory and MSD figures from stage simulation manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_dump(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timesteps: List[int] = []
    ids_reference: np.ndarray | None = None
    frames: List[np.ndarray] = []

    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                continue

            timestep = int(handle.readline().strip())
            number_header = handle.readline().strip()
            if number_header != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Unexpected dump structure in {path}: {number_header}")
            n_atoms = int(handle.readline().strip())

            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump structure in {path}: missing box bounds")
            handle.readline()
            handle.readline()
            handle.readline()

            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected dump structure in {path}: missing atom header")
            columns = atoms_header.split()[2:]
            id_col = columns.index("id")
            x_col = columns.index("xu") if "xu" in columns else columns.index("x")
            y_col = columns.index("yu") if "yu" in columns else columns.index("y")

            ids = np.empty(n_atoms, dtype=int)
            xy = np.empty((n_atoms, 2), dtype=float)
            for index in range(n_atoms):
                parts = handle.readline().split()
                ids[index] = int(parts[id_col])
                xy[index, 0] = float(parts[x_col])
                xy[index, 1] = float(parts[y_col])

            order = np.argsort(ids)
            ids = ids[order]
            xy = xy[order]
            if ids_reference is None:
                ids_reference = ids
            elif not np.array_equal(ids_reference, ids):
                raise ValueError(f"Atom IDs changed order/content in {path}")

            timesteps.append(timestep)
            frames.append(xy)

    if not frames or ids_reference is None:
        raise ValueError(f"No frames found in {path}")
    return np.asarray(timesteps, dtype=int), ids_reference, np.stack(frames, axis=0)


def make_lag_indices(n_frames: int, n_lags: int = 160, all_lags: bool = False) -> np.ndarray:
    max_lag = max(1, n_frames // 2)
    if all_lags or max_lag <= n_lags:
        return np.arange(1, max_lag + 1, dtype=int)
    lags = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))
    return lags[lags > 0]


def displacement_statistics(
    timesteps: np.ndarray,
    positions: np.ndarray,
    timestep_size: float,
    all_lags: bool,
) -> Dict[str, np.ndarray]:
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    lags = make_lag_indices(len(timesteps), all_lags=all_lags)

    rows: Dict[str, List[float]] = {
        "lag_index": [],
        "lag_steps": [],
        "lag_time": [],
        "n_samples": [],
        "mean_dx": [],
        "sem_dx": [],
        "raw_msd_x": [],
        "centered_msd_x": [],
        "raw_msd_y": [],
    }

    for lag in lags:
        dx = (x[lag:] - x[:-lag]).reshape(-1)
        dy = (y[lag:] - y[:-lag]).reshape(-1)
        lag_steps = int(np.median(timesteps[lag:] - timesteps[:-lag]))
        mean_dx = float(np.mean(dx))
        centered = dx - mean_dx
        n_samples = dx.size

        rows["lag_index"].append(lag)
        rows["lag_steps"].append(lag_steps)
        rows["lag_time"].append(lag_steps * timestep_size)
        rows["n_samples"].append(n_samples)
        rows["mean_dx"].append(mean_dx)
        rows["sem_dx"].append(float(np.std(dx, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else 0.0)
        rows["raw_msd_x"].append(float(np.mean(dx**2)))
        rows["centered_msd_x"].append(float(np.mean(centered**2)))
        rows["raw_msd_y"].append(float(np.mean(dy**2)))

    return {key: np.asarray(value) for key, value in rows.items()}


def fit_line(x: np.ndarray, y: np.ndarray, through_origin: bool = False) -> Tuple[float, float]:
    if through_origin:
        slope = float(np.dot(x, y) / np.dot(x, x))
        return slope, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def fit_power_law(tau: np.ndarray, msd: np.ndarray) -> Tuple[float, float]:
    mask = (tau > 0) & (msd > 0)
    alpha, log_amp = np.polyfit(np.log(tau[mask]), np.log(msd[mask]), 1)
    return float(np.exp(log_amp)), float(alpha)


def fit_mask(lag_time: np.ndarray) -> np.ndarray:
    positive = lag_time > 0
    if np.count_nonzero(positive) < 4:
        return positive
    upper = np.quantile(lag_time[positive], 0.4)
    return positive & (lag_time <= upper)


def save_msd_csv(path: Path, stats: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(stats.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for index in range(len(stats[keys[0]])):
            writer.writerow([stats[key][index] for key in keys])


def plot_trajectory(path: Path, times: np.ndarray, positions: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = positions[:, :, 0]
    for atom_index in range(x.shape[1]):
        ax.plot(times, x[:, atom_index], lw=1.3 if x.shape[1] <= 5 else 0.7, alpha=0.85)
    ax.set_xlabel("time")
    ax.set_ylabel("unwrapped x")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_diffusion_msd(
    path: Path,
    stats: Dict[str, np.ndarray],
    d_fit: float,
    amp: float,
    alpha: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tau = stats["lag_time"]
    raw = stats["raw_msd_x"]
    centered = stats["centered_msd_x"]
    mask = tau > 0

    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    ax.loglog(tau[mask], raw[mask], "o", ms=4, alpha=0.75, label="raw MSD x")
    ax.loglog(tau[mask], centered[mask], "s", ms=3.5, alpha=0.55, label="centered MSD x")
    ax.loglog(tau[mask], 2.0 * d_fit * tau[mask], "-", lw=2, label=f"2Dt, D={d_fit:.3g}")
    ax.loglog(tau[mask], amp * tau[mask] ** alpha, "--", lw=2, label=f"A t^alpha, alpha={alpha:.3g}")
    ax.set_xlabel("lag time")
    ax.set_ylabel("MSD x")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_force_summary(
    mean_path: Path,
    msd_path: Path,
    stats: Dict[str, np.ndarray],
    v_fit: float,
    d_fit: float,
    title: str,
) -> None:
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    tau = stats["lag_time"]
    mask = tau > 0

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.plot(tau[mask], stats["mean_dx"][mask], "o", ms=4, alpha=0.75, label="mean dx")
    ax.plot(tau[mask], v_fit * tau[mask], "-", lw=2, label=f"v t, v={v_fit:.3g}")
    ax.set_xlabel("lag time")
    ax.set_ylabel("mean displacement x")
    ax.set_title(title)
    ax.legend()
    fig.savefig(mean_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    ax.loglog(tau[mask], stats["raw_msd_x"][mask], "o", ms=4, alpha=0.75, label="raw MSD x")
    ax.loglog(tau[mask], stats["centered_msd_x"][mask], "s", ms=4, alpha=0.75, label="centered MSD x")
    ax.loglog(tau[mask], 2.0 * d_fit * tau[mask], "-", lw=2, label=f"2Dt, D={d_fit:.3g}")
    ax.set_xlabel("lag time")
    ax.set_ylabel("MSD x")
    ax.set_title(title)
    ax.legend()
    fig.savefig(msd_path, dpi=180)
    plt.close(fig)


def analyze_row(row: Dict[str, str], analyze_dir: Path, figures_dir: Path, all_lags: bool) -> Dict[str, float | str]:
    dump_path = Path(row["dump_file"])
    stem = dump_path.stem
    timestep_size = float(row["timestep"])
    mode = row["mode"]
    timesteps, _ids, positions = parse_dump(dump_path)
    times = timesteps * timestep_size
    stats = displacement_statistics(timesteps, positions, timestep_size, all_lags=all_lags)

    save_msd_csv(analyze_dir / f"{stem}_msd.csv", stats)
    plot_trajectory(figures_dir / f"{stem}_trajectory.png", times, positions, f"{stem}: x trajectory")

    tau = stats["lag_time"]
    mask = fit_mask(tau)
    centered = stats["centered_msd_x"]
    raw = stats["raw_msd_x"]

    d_slope, _ = fit_line(tau[mask], centered[mask], through_origin=True)
    d_fit = 0.5 * d_slope
    amp, alpha = fit_power_law(tau[mask], centered[mask])

    summary: Dict[str, float | str] = {
        "stem": stem,
        "mode": mode,
        "N": int(row["N"]),
        "width_factor": float(row["width_factor"]),
        "run_id": int(row["run_id"]),
        "n_frames": len(timesteps),
        "total_time": float(times[-1] - times[0]),
        "D_centered_fit": d_fit,
        "power_amp": amp,
        "power_alpha": alpha,
    }

    if mode == "force":
        v_fit, v_intercept = fit_line(tau[mask], stats["mean_dx"][mask], through_origin=False)
        summary["v_fit"] = v_fit
        summary["v_intercept"] = v_intercept
        plot_force_summary(
            figures_dir / f"{stem}_mean_displacement.png",
            figures_dir / f"{stem}_raw_vs_centered_msd.png",
            stats,
            v_fit,
            d_fit,
            f"{stem}: force validation",
        )
    else:
        raw_d_slope, _ = fit_line(tau[mask], raw[mask], through_origin=True)
        summary["D_raw_fit"] = 0.5 * raw_d_slope
        plot_diffusion_msd(
            figures_dir / f"{stem}_msd.png",
            stats,
            d_fit,
            amp,
            alpha,
            f"{stem}: diffusion validation",
        )

    with (analyze_dir / f"{stem}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_summary(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--analyze-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--all-lags", action="store_true")
    args = parser.parse_args()

    rows = [row for row in read_manifest(args.manifest) if row.get("status") == "success"]
    if not rows:
        raise SystemExit(f"No successful simulations found in {args.manifest}")

    args.analyze_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = [analyze_row(row, args.analyze_dir, args.figures_dir, args.all_lags) for row in rows]
    write_summary(args.analyze_dir / "summary_by_run.csv", summaries)
    print(f"Wrote {len(summaries)} analyzed run(s)")
    print(f"Figures: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
