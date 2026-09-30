"""Shared configuration and derived quantities for stage simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Dict


@dataclass(frozen=True)
class SimulationDefaults:
    length: float = 1.0
    sigma: float = 0.01
    diffusion_coef: float = 0.01
    temperature: float = 1.0
    timestep: float = 1e-6
    runtime_factor: float = 2e3
    target_frames: int = 2000
    walk_velocity: float = 1.0
    min_runsteps: int = 1000
    collision_offset: float = 1e-3


DEFAULTS = SimulationDefaults()


def estimate_collision_time(
    n_particles: int,
    length: float = DEFAULTS.length,
    sigma: float = DEFAULTS.sigma,
    diffusion_coef: float = DEFAULTS.diffusion_coef,
    offset: float = DEFAULTS.collision_offset,
) -> float:
    """Return the Brownian time for a particle to explore the mean gap.

    For N=1 there is no inter-particle collision time, so use the time to
    diffuse over one diameter as the smallest meaningful Brownian scale.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    if diffusion_coef <= 0:
        raise ValueError("diffusion_coef must be positive")

    if n_particles == 1:
        return sigma**2 / diffusion_coef + offset

    gap = max(length / n_particles - sigma, 0.0)
    return gap**2 / diffusion_coef + offset


def derived_parameters(
    n_particles: int,
    width_factor: float,
    mode: str,
    defaults: SimulationDefaults = DEFAULTS,
    runtime_factor: float | None = None,
    target_frames: int | None = None,
) -> Dict[str, Any]:
    """Compute the values passed to LAMMPS and recorded in manifests."""
    if mode not in {"diffusion", "force"}:
        raise ValueError("mode must be 'diffusion' or 'force'")
    if width_factor <= 0:
        raise ValueError("width_factor must be positive")

    factor = defaults.runtime_factor if runtime_factor is None else runtime_factor
    frames = defaults.target_frames if target_frames is None else target_frames
    if factor <= 0:
        raise ValueError("runtime_factor must be positive")
    if frames < 2:
        raise ValueError("target_frames must be >= 2")

    t_coll = estimate_collision_time(
        n_particles=n_particles,
        length=defaults.length,
        sigma=defaults.sigma,
        diffusion_coef=defaults.diffusion_coef,
        offset=defaults.collision_offset,
    )
    t_sigma = defaults.sigma**2 / defaults.diffusion_coef
    run_time = factor * max(t_coll, t_sigma)
    runsteps = max(defaults.min_runsteps, int(ceil(run_time / defaults.timestep)))
    dumpfreq = max(1, int(ceil(runsteps / frames)))
    gamma = defaults.temperature / defaults.diffusion_coef

    values: Dict[str, Any] = {
        **asdict(defaults),
        "mode": mode,
        "N": n_particles,
        "width_factor": width_factor,
        "gamma": gamma,
        "t_coll": t_coll,
        "t_sigma": t_sigma,
        "run_time": run_time,
        "runsteps": runsteps,
        "dumpfreq": dumpfreq,
        "expected_v": defaults.walk_velocity if mode == "force" else 0.0,
        "runtime_factor": factor,
        "target_frames": frames,
    }
    return values
