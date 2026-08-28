#!/usr/bin/env python3
"""Validate and submit the nighthawk rotated-60-degree QAOA experiment.

This script expects the directory produced by prepare_nighthawk_curved60rot45_qpu_preflight.py.
Without --submit it performs validation only and writes a preview JSON.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from step11_common import gate_mode_to_fractional

# PATCHED_FOR_ARBITRARY_SEED_COHORT_V1


def load_qpy_list(path: Path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def backend_name(backend) -> str:
    value = getattr(backend, "name", "unknown")
    return str(value() if callable(value) else value)


def status_payload(status) -> dict:
    return {
        "operational": bool(getattr(status, "operational", False)),
        "pending_jobs": int(getattr(status, "pending_jobs", -1)),
        "status_message": str(getattr(status, "status_msg", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--shots-per-pub", type=int, default=10_000)
    parser.add_argument(
        "--dd", choices=["none", "XX", "XpXm", "XY4"], default="XY4"
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.shots_per_pub < 1:
        raise ValueError("--shots-per-pub must be positive")

    prepared = args.prepared
    validation_path = prepared / "nighthawk_curved60rot45_preflight_validation.json"
    validation = json.loads(validation_path.read_text())

    if not bool(validation.get("submission_allowed", False)):
        raise RuntimeError("Preflight does not permit submission")
    if not bool(validation.get("all_seeds_pass_calibration_safety", False)):
        raise RuntimeError("Not all selected seeds passed calibration safety")

    files = validation["files"]
    candidates_path = prepared / files["candidates"]
    qpy_path = prepared / files["qpy"]
    candidates = pd.read_csv(candidates_path)
    circuits = load_qpy_list(qpy_path)

    if len(circuits) != len(candidates):
        raise RuntimeError(
            f"QPY/candidate mismatch: {len(circuits)} circuits and "
            f"{len(candidates)} candidate rows"
        )
    if len(circuits) != int(validation["n_prepared_pubs"]):
        raise RuntimeError("Prepared PUB count differs from validation JSON")

    expected_indices = list(range(len(candidates)))
    observed_indices = candidates["pub_index"].astype(int).tolist()
    if observed_indices != expected_indices:
        raise RuntimeError(
            f"Candidate pub_index must be sequential {expected_indices}; "
            f"found {observed_indices}"
        )

    backend_requested = str(validation["backend_status"]["backend"])
    gate_mode = str(validation["gate_mode"])
    use_fractional = gate_mode_to_fractional(gate_mode)

    service = QiskitRuntimeService()
    backend = service.backend(
        backend_requested,
        use_fractional_gates=use_fractional,
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f"Backend {backend_requested} is not operational: {status}")

    operations = set(map(str, backend.operation_names))
    if gate_mode == "fractional" and "rzz" not in operations:
        raise RuntimeError("Fractional backend target no longer exposes rzz")

    for index, circuit in enumerate(circuits):
        backend.check_faulty(circuit)
        if circuit.num_clbits != 50:
            raise RuntimeError(
                f"PUB {index}: expected 50 classical bits; found {circuit.num_clbits}"
            )
        if circuit.parameters:
            raise RuntimeError(f"PUB {index}: circuit still has unbound parameters")

    preview = {
        **validation,
        "validated_immediately_before_submission_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "backend_live": {
            "backend": backend_name(backend),
            **status_payload(status),
        },
        "shots_per_pub": int(args.shots_per_pub),
        "total_requested_shots": int(args.shots_per_pub * len(circuits)),
        "dd": args.dd,
        "gate_twirling": False,
        "measurement_twirling": False,
        "execution_mode": f"{len(circuits)}_bound_folded_pubs_in_one_runtime_job",
        "files": {
            **files,
            "validation": validation_path.name,
        },
    }
    preview_path = prepared / "nighthawk_curved60rot45_submission_preview.json"
    preview_path.write_text(json.dumps(preview, indent=2, default=str))

    print(json.dumps(preview, indent=2, default=str), flush=True)
    if not args.submit:
        print(
            "\nValidation only. Re-run with --submit to use QPU time.",
            flush=True,
        )
        return

    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    if args.dd == "none":
        sampler.options.dynamical_decoupling.enable = False
    else:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = args.dd
    sampler.options.environment.job_tags = [
        "DMRI_CROSSING_QAOA",
        "nighthawk_rotated_60",
        f"seeds_{'_'.join(str(seed) for seed in validation['seeds'])}",
        gate_mode,
        f"dd_{args.dd}",
        "standard_cz_untwirled",
    ]

    job = sampler.run(
        [(circuit,) for circuit in circuits],
        shots=int(args.shots_per_pub),
    )
    submission = {
        **preview,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id(),
    }
    submission_path = prepared / "nighthawk_curved60rot45_submission.json"
    submission_path.write_text(json.dumps(submission, indent=2, default=str))

    print(f"\nSubmitted Runtime job: {job.job_id()}", flush=True)
    print(f"Submission record: {submission_path}", flush=True)


if __name__ == "__main__":
    main()
