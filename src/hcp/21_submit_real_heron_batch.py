#!/usr/bin/env python3
"""Validate, preview, and optionally submit 30 real-data Heron QAOA circuits.

The script scans:
    real_data_replication/<patch_id>/heron_preflight/

It requires one successful, calibration-aware preflight and one winning QPY
circuit per patch. Before submission, it performs live checks against the
requested IBM backend:

  * backend is operational;
  * fractional-gate mode matches the preflight;
  * the live calibration timestamp matches every preflight timestamp;
  * each QPY contains exactly one bound circuit;
  * each circuit has 50 classical output bits;
  * every circuit passes backend.check_faulty();
  * all preflight safety checks remain valid.

Without --submit, only a preview is written. With --submit, all circuits are
submitted as PUBs in one SamplerV2 Runtime job with identical execution
settings.

Primary protocol:
    backend: ibm_aachen
    gate mode: fractional
    angle strategy: hardware_calibrated
    shots: 10,000 per PUB
    DD: XY4
    gate twirling: off
    measurement twirling: off
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("real_data_replication"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("real_heron_submission"),
    )
    parser.add_argument("--backend", default="ibm_aachen")
    parser.add_argument("--expected-patches", type=int, default=30)
    parser.add_argument("--shots-per-pub", type=int, default=10_000)
    parser.add_argument(
        "--dd",
        choices=["none", "XX", "XpXm", "XY4"],
        default="XY4",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the one-job 30-PUB Runtime experiment.",
    )
    return parser.parse_args()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def backend_name(backend: Any) -> str:
    value = getattr(backend, "name", "unknown")
    return str(value() if callable(value) else value)


def status_payload(status: Any) -> dict[str, Any]:
    return {
        "operational": bool(getattr(status, "operational", False)),
        "pending_jobs": int(getattr(status, "pending_jobs", -1)),
        "status_message": str(getattr(status, "status_msg", "")),
    }


def calibration_timestamp(backend: Any) -> str | None:
    try:
        properties = backend.properties()
        value = getattr(properties, "last_update_date", None)
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    except Exception:
        return None


def timestamps_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left == right
    try:
        return bool(pd.Timestamp(left) == pd.Timestamp(right))
    except Exception:
        return str(left) == str(right)


def load_single_qpy(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != 1:
        raise RuntimeError(
            f"Expected one circuit in {path}; found {len(circuits)}"
        )
    return circuits[0]


def require_safety(validation: dict[str, Any], patch_id: str) -> None:
    safety = validation.get("safety", {})
    requirements = {
        "winning_max_twoq_error_within_threshold": True,
        "winning_active_qubits_fit_backend": True,
        "twoq_depth_warning": False,
    }
    for field, expected in requirements.items():
        observed = safety.get(field)
        if observed is not expected:
            raise RuntimeError(
                f"{patch_id}: safety field {field}={observed!r}; "
                f"expected {expected!r}"
            )

    winner = validation.get("selection", {}).get("winner", {})
    if int(winner.get("n_twoq_errors_missing", 1)) != 0:
        raise RuntimeError(
            f"{patch_id}: used two-qubit calibration errors are missing"
        )


def main() -> None:
    args = parse_args()
    if args.expected_patches < 1:
        raise ValueError("--expected-patches must be positive")
    if args.shots_per_pub < 1:
        raise ValueError("--shots-per-pub must be positive")

    args.root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()

    validation_paths = sorted(
        args.root.glob("*/heron_preflight/real_qpu_preflight.json")
    )
    if len(validation_paths) != args.expected_patches:
        raise RuntimeError(
            f"Expected {args.expected_patches} successful preflights; "
            f"found {len(validation_paths)}"
        )

    records: list[dict[str, Any]] = []
    circuits = []
    preflight_calibrations: list[str | None] = []

    for pub_index, validation_path in enumerate(validation_paths):
        patch_dir = validation_path.parents[1]
        patch_id = patch_dir.name
        preflight_dir = validation_path.parent
        validation = json.loads(
            validation_path.read_text(encoding="utf-8")
        )

        require_safety(validation, patch_id)

        backend_record = validation["backend"]
        if str(backend_record["name"]) != args.backend:
            raise RuntimeError(
                f"{patch_id}: backend {backend_record['name']} "
                f"does not match {args.backend}"
            )
        if not bool(backend_record.get("use_fractional_gates")):
            raise RuntimeError(
                f"{patch_id}: winning circuit is not in fractional-gate mode"
            )

        circuit_record = validation["circuit"]
        if str(circuit_record["angle_strategy"]) != "hardware_calibrated":
            raise RuntimeError(
                f"{patch_id}: unexpected angle strategy "
                f"{circuit_record['angle_strategy']}"
            )

        qpy_name = str(validation["files"]["qpy"])
        qpy_path = preflight_dir / qpy_name
        circuit = load_single_qpy(qpy_path)

        if circuit.parameters:
            raise RuntimeError(
                f"{patch_id}: unbound parameters remain: {circuit.parameters}"
            )
        if int(circuit.num_clbits) != 50:
            raise RuntimeError(
                f"{patch_id}: expected 50 classical bits, "
                f"found {circuit.num_clbits}"
            )

        winner = validation["selection"]["winner"]
        prepared_calibration = backend_record.get(
            "calibration_last_update"
        )
        preflight_calibrations.append(prepared_calibration)

        records.append(
            {
                "pub_index": int(pub_index),
                "patch_id": patch_id,
                "patch_dir": str(patch_dir.resolve()),
                "graph_dir": str(
                    (patch_dir / "graph_validation").resolve()
                ),
                "mps_results_csv": str(
                    (
                        patch_dir
                        / "mps_ideal"
                        / "real_mps_results.csv"
                    ).resolve()
                ),
                "preflight_dir": str(preflight_dir.resolve()),
                "preflight_json": str(validation_path.resolve()),
                "source_qpy": str(qpy_path.resolve()),
                "backend": str(backend_record["name"]),
                "prepared_calibration_last_update": (
                    prepared_calibration
                ),
                "angle_strategy": str(
                    circuit_record["angle_strategy"]
                ),
                "gamma": float(circuit_record["gamma"]),
                "beta": float(circuit_record["beta"]),
                "transpiler_seed": int(
                    validation["selection"]["winning_seed"]
                ),
                "depth": int(winner["depth"]),
                "twoq_depth": int(winner["twoq_depth"]),
                "n_twoq_gates": int(winner["n_twoq_gates"]),
                "n_active_physical_qubits": int(
                    winner["n_active_physical_qubits"]
                ),
                "mean_twoq_error": float(winner["mean_twoq_error"]),
                "max_twoq_error": float(winner["max_twoq_error"]),
                "twoq_calibration_error_cost": float(
                    winner["twoq_calibration_error_cost"]
                ),
                "independent_twoq_success_proxy": float(
                    winner["independent_twoq_success_proxy"]
                ),
            }
        )
        circuits.append(circuit)

    unique_calibrations = {
        str(value)
        for value in preflight_calibrations
        if value is not None
    }
    if len(unique_calibrations) != 1:
        raise RuntimeError(
            "The circuits were not all transpiled against one calibration "
            f"snapshot: {sorted(unique_calibrations)}"
        )

    service = QiskitRuntimeService()
    backend = service.backend(
        args.backend,
        use_fractional_gates=True,
    )
    if backend_name(backend) != args.backend:
        raise RuntimeError(
            f"Requested {args.backend}, received {backend_name(backend)}"
        )

    live_status = status_payload(backend.status())
    if not live_status["operational"]:
        raise RuntimeError(
            f"Backend {args.backend} is not operational: {live_status}"
        )

    live_calibration = calibration_timestamp(backend)
    for record, prepared_calibration, circuit in zip(
        records,
        preflight_calibrations,
        circuits,
        strict=True,
    ):
        patch_id = str(record["patch_id"])
        if not timestamps_equal(prepared_calibration, live_calibration):
            raise RuntimeError(
                "Backend calibration changed after transpilation.\n"
                f"Patch:                {patch_id}\n"
                f"Prepared calibration: {prepared_calibration}\n"
                f"Live calibration:     {live_calibration}\n"
                "Re-run the 30-case Heron preflight before submission."
            )

        backend.check_faulty(circuit)

    candidate_frame = pd.DataFrame(records).sort_values(
        "pub_index"
    ).reset_index(drop=True)

    path_checks = {
        "graph_dir": "directory",
        "mps_results_csv": "file",
        "source_qpy": "file",
    }
    for path_column, expected_type in path_checks.items():
        missing = []
        for value in candidate_frame[path_column]:
            path = Path(str(value))
            exists = (
                path.is_dir()
                if expected_type == "directory"
                else path.is_file()
            )
            if not exists:
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                f"Missing {path_column} {expected_type}s: {missing[:5]}"
            )

    candidates_path = args.out / "real_heron_submission_candidates.csv"
    candidate_frame.to_csv(candidates_path, index=False)

    combined_qpy_path = args.out / "real_heron_30pubs.qpy"
    with combined_qpy_path.open("wb") as handle:
        qpy.dump(circuits, handle)

    pub_order = candidate_frame["patch_id"].astype(str).tolist()
    preview = {
        "experiment": "real_HCP_YA_30_subject_QAOA",
        "validated_immediately_before_submission_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "backend_live": {
            "backend": args.backend,
            **live_status,
            "calibration_last_update": live_calibration,
            "use_fractional_gates": True,
        },
        "protocol": {
            "primitive": "SamplerV2",
            "execution_mode": "one_runtime_job",
            "n_pubs": int(len(circuits)),
            "shots_per_pub": int(args.shots_per_pub),
            "total_requested_shots": int(
                len(circuits) * args.shots_per_pub
            ),
            "angle_strategy": "hardware_calibrated",
            "gate_mode": "fractional",
            "dynamical_decoupling": args.dd,
            "gate_twirling": False,
            "measurement_twirling": False,
        },
        "pub_order": pub_order,
        "preflight_summary": {
            "n_circuits": int(len(circuits)),
            "prepared_calibration_last_update": next(
                iter(unique_calibrations)
            ),
            "minimum_twoq_depth": int(
                candidate_frame["twoq_depth"].min()
            ),
            "median_twoq_depth": float(
                candidate_frame["twoq_depth"].median()
            ),
            "maximum_twoq_depth": int(
                candidate_frame["twoq_depth"].max()
            ),
            "maximum_used_twoq_error": float(
                candidate_frame["max_twoq_error"].max()
            ),
            "minimum_independent_twoq_success_proxy": float(
                candidate_frame[
                    "independent_twoq_success_proxy"
                ].min()
            ),
        },
        "files": {
            "candidates": candidates_path.name,
            "combined_qpy": combined_qpy_path.name,
        },
        "safety": {
            "expected_patch_count_present": True,
            "all_preflights_passed": True,
            "one_calibration_snapshot": True,
            "backend_operational": True,
            "calibration_unchanged": True,
            "backend_check_faulty_passed_for_all_circuits": True,
            "qpu_job_submitted": False,
        },
    }

    preview_path = args.out / "real_heron_submission_preview.json"
    preview_path.write_text(
        json.dumps(preview, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(preview, indent=2, default=json_default), flush=True)

    if not args.submit:
        print(
            "\nValidation only. No QPU job was submitted.\n"
            "Review the preview, then rerun the same command with --submit.",
            flush=True,
        )
        return

    submission_path = args.out / "real_heron_submission.json"
    if submission_path.exists():
        raise RuntimeError(
            f"{submission_path} already exists. Refusing a possible "
            "duplicate submission."
        )

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
        "real_HCP_YA_30_subjects",
        "right_CST_SCR",
        "hardware_calibrated",
        "fractional_gates",
        f"dd_{args.dd}",
        "untwirled",
    ]

    job = sampler.run(
        [(circuit,) for circuit in circuits],
        shots=int(args.shots_per_pub),
    )

    submission = {
        **preview,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id(),
        "safety": {
            **preview["safety"],
            "qpu_job_submitted": True,
        },
    }
    submission_path.write_text(
        json.dumps(submission, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print(f"\nSubmitted Runtime job: {job.job_id()}", flush=True)
    print(f"Submission record: {submission_path}", flush=True)


if __name__ == "__main__":
    main()
