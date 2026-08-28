#!/usr/bin/env python3
"""Generate four held-out Nighthawk phantom workflows from existing Heron templates.

Run from crossing_fibres_step7. The script reads:
  prepare_rotated90_qpu_preflight.py
  submit_rotated90_qpu.py
  collect_rotated90_qpu.py

It writes new Nighthawk files only and does not modify the Heron scripts.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


SOURCE_FILES = {
    "prepare": Path("prepare_rotated90_qpu_preflight.py"),
    "submit": Path("submit_rotated90_qpu.py"),
    "collect": Path("collect_rotated90_qpu.py"),
}

GAMMA_NIGHTHAWK = 0.7071731744054941
BETA_NIGHTHAWK = 1.1743489498936892
GAMMA_TRANSFERRED = 0.7771731744054942
BETA_TRANSFERRED = 1.1743489498936892

CONFIGURATIONS = {
    "nighthawk_curved90": {
        "geometry": "curved_90",
        "experiment": "nighthawk_curved_90_degree_phantom",
        "job_tag": "nighthawk_curved_90",
        "description": "Nighthawk curved-90-degree",
    },
    "nighthawk_rotated90": {
        "geometry": "curved_90_rot45",
        "experiment": "nighthawk_rotated_90_degree_orientation_generalization",
        "job_tag": "nighthawk_rotated_90",
        "description": "Nighthawk rotated-90-degree",
    },
    "nighthawk_curved60": {
        "geometry": "curved_60",
        "experiment": "nighthawk_curved_60_degree_phantom",
        "job_tag": "nighthawk_curved_60",
        "description": "Nighthawk curved-60-degree",
    },
    "nighthawk_curved60rot45": {
        "geometry": "curved_60_rot45",
        "experiment": "nighthawk_rotated_60_degree_orientation_generalization",
        "job_tag": "nighthawk_rotated_60",
        "description": "Nighthawk rotated-60-degree",
    },
}


ANGLE_BLOCK = f"""ANGLE_STRATEGIES = {{
    "nighthawk_calibrated": {{
        "gamma": {GAMMA_NIGHTHAWK!r},
        "beta": {BETA_NIGHTHAWK!r},
    }},
    "transferred_center": {{
        "gamma": {GAMMA_TRANSFERRED!r},
        "beta": {BETA_TRANSFERRED!r},
    }},
}}"""


BIND_STANDARD_FUNCTION = """def bind_standard(parameterized_isa, *, gamma: float, beta: float, backend):
    \"\"\"Bind an already-routed standard-CZ ISA circuit without fractional folding.\"\"\"
    parameters = {
        parameter.name: parameter
        for parameter in parameterized_isa.parameters
    }
    if set(parameters) != {"gamma", "beta"}:
        raise ValueError(
            f"Expected gamma and beta; found {sorted(parameters)}"
        )

    bound = parameterized_isa.assign_parameters(
        {
            parameters["gamma"]: float(gamma),
            parameters["beta"]: float(beta),
        },
        inplace=False,
    )
    if bound.parameters:
        raise RuntimeError(
            f"Unbound parameters remain: {sorted(p.name for p in bound.parameters)}"
        )

    backend.check_faulty(bound)
    supported = set(map(str, backend.operation_names))
    operations = {
        str(name): int(count)
        for name, count in bound.count_ops().items()
    }
    unsupported = sorted(
        name
        for name in operations
        if name not in supported and name not in {"barrier"}
    )
    if unsupported:
        raise RuntimeError(
            f"Bound standard circuit contains non-ISA operations: {unsupported}"
        )
    if int(operations.get("rzz", 0)) != 0:
        raise RuntimeError(
            "Standard Nighthawk circuit unexpectedly contains RZZ gates"
        )

    return bound, []


"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def patch_prepare(text: str) -> str:
    text = text.replace(
        "from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle\n",
        "",
    )

    text = re.sub(
        r"DEFAULT_SEEDS\s*=\s*\([^\n]*\)",
        "DEFAULT_SEEDS = tuple(range(2001, 2016))",
        text,
        count=1,
    )

    angle_pattern = re.compile(
        r'ANGLE_STRATEGIES\s*=\s*\{\n.*?\n\}\n\n',
        flags=re.DOTALL,
    )
    text, n = angle_pattern.subn(ANGLE_BLOCK + "\n\n", text, count=1)
    if n != 1:
        raise RuntimeError("Could not replace ANGLE_STRATEGIES")

    function_pattern = re.compile(
        r"def bind_and_fold\(.*?\n\n(?=def trial_sort_key)",
        flags=re.DOTALL,
    )
    text, n = function_pattern.subn(BIND_STANDARD_FUNCTION, text, count=1)
    if n != 1:
        raise RuntimeError("Could not replace bind_and_fold")

    text = text.replace("bind_and_fold(", "bind_standard(")
    text = text.replace(
        'ANGLE_STRATEGIES["hardware_calibrated"]',
        'ANGLE_STRATEGIES["nighthawk_calibrated"]',
    )
    text = text.replace('"hardware_calibrated"', '"nighthawk_calibrated"')
    text = text.replace("hardware-calibrated", "Nighthawk-calibrated")

    text = text.replace(
        'parser.add_argument("--backend", default="ibm_aachen")',
        'parser.add_argument("--backend", default="ibm_berlin")',
    )
    text = text.replace(
        '--gate-mode", choices=["standard", "fractional"], default="fractional"',
        '--gate-mode", choices=["standard", "fractional"], default="standard"',
    )

    old_error_lookup = """        mapping = backend.target[operation_name]
        properties = mapping.get(tuple(qargs))
        if properties is None or properties.error is None:
            return None
"""
    new_error_lookup = """        mapping = backend.target[operation_name]
        properties = mapping.get(tuple(qargs))
        if properties is None and len(qargs) == 2:
            properties = mapping.get(tuple(reversed(qargs)))
        if properties is None or properties.error is None:
            return None
"""
    text = replace_once(
        text,
        old_error_lookup,
        new_error_lookup,
        "target_instruction_error",
    )

    text = text.replace("binds and folds", "binds")
    text = text.replace("bound_folded", "bound_standard")
    text = text.replace("Bound and folded", "Bound standard")
    return text


def patch_submit(text: str) -> str:
    text = text.replace(
        "six_bound_folded_pubs_in_one_runtime_job",
        "bound_standard_pubs_in_one_runtime_job",
    )
    text = text.replace("seeds_2003_2004_2011", "seeds_2001_2015")
    text = text.replace("bound_folded_rzz", "standard_cz_untwirled")
    return text


def patch_collect(text: str) -> str:
    text = text.replace('"hardware_calibrated"', '"nighthawk_calibrated"')
    text = text.replace("hardware-calibrated", "Nighthawk-calibrated")
    text = text.replace(
        "Paired hardware-calibrated minus transferred comparison:",
        "Paired Nighthawk-calibrated minus transferred comparison:",
    )
    return text


def geometry_transform(
    text: str,
    *,
    prefix: str,
    config: dict[str, str],
) -> str:
    replacements = [
        (
            "rotated_90_degree_orientation_generalization",
            config["experiment"],
        ),
        ("rotated_90_generalization", config["job_tag"]),
        ("curved_90_rot45", config["geometry"]),
        ("rotated90", prefix),
        ("Rotated-90-degree", config["description"]),
        ("rotated-90-degree", config["description"].lower()),
        ("Rotated-90", config["description"]),
        ("rotated-90", config["description"].lower()),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def validate(
    *,
    role: str,
    path: Path,
    text: str,
    prefix: str,
    geometry: str,
) -> None:
    ast.parse(text, filename=str(path))

    # Strategy names are embedded in the prepared candidate table and are
    # therefore required in the prepare and collect scripts. The submit script
    # only validates/submits the prepared circuits and need not contain them.
    required_tokens = [prefix]
    if role in {"prepare", "collect"}:
        required_tokens.extend(
            ["nighthawk_calibrated", "transferred_center"]
        )

    for token in required_tokens:
        if token not in text:
            raise RuntimeError(f"{path}: missing required token {token!r}")

    for token in ('"hardware_calibrated"', "seeds_2003_2004_2011"):
        if token in text:
            raise RuntimeError(f"{path}: stale token remains: {token!r}")

    if role == "prepare":
        for token in ("FoldRzzAngle", "bind_and_fold("):
            if token in text:
                raise RuntimeError(f"{path}: stale token remains: {token!r}")
        for token in (
            geometry,
            "tuple(range(2001, 2016))",
            'default="ibm_berlin"',
            'default="standard"',
            "bind_standard(",
        ):
            if token not in text:
                raise RuntimeError(f"{path}: missing {token!r}")

    if role == "submit":
        # The submitter consumes the prepared validation/candidate metadata.
        # It therefore need not repeat the seed range or strategy labels.
        # Only the execution-safety options must be present literally.
        for token in (
            '"gate_twirling": False',
            '"measurement_twirling": False',
        ):
            if token not in text:
                raise RuntimeError(f"{path}: missing {token!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace previously generated Nighthawk files.",
    )
    args = parser.parse_args()

    missing = [str(path) for path in SOURCE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run from crossing_fibres_step7. Missing templates: "
            + ", ".join(missing)
        )

    sources = {
        role: path.read_text(encoding="utf-8")
        for role, path in SOURCE_FILES.items()
    }
    base = {
        "prepare": patch_prepare(sources["prepare"]),
        "submit": patch_submit(sources["submit"]),
        "collect": patch_collect(sources["collect"]),
    }

    planned = []
    for prefix, config in CONFIGURATIONS.items():
        targets = {
            "prepare": Path(f"prepare_{prefix}_qpu_preflight.py"),
            "submit": Path(f"submit_{prefix}_qpu.py"),
            "collect": Path(f"collect_{prefix}_qpu.py"),
        }
        for role, target in targets.items():
            if target.exists() and not args.force:
                raise FileExistsError(
                    f"{target} already exists. Use --force only to replace it."
                )
            text = geometry_transform(
                base[role],
                prefix=prefix,
                config=config,
            )
            validate(
                role=role,
                path=target,
                text=text,
                prefix=prefix,
                geometry=config["geometry"],
            )
            planned.append((target, text))

    for target, text in planned:
        target.write_text(text, encoding="utf-8")
        target.chmod(0o755)

    print("Generated and syntax-checked:")
    for target, _ in planned:
        print(f"  {target}")

    seeds = " ".join(str(seed) for seed in range(2001, 2016))
    print("\nPreflight commands only — no QPU execution:")
    for prefix in CONFIGURATIONS:
        print(
            f"python prepare_{prefix}_qpu_preflight.py "
            f"--backend ibm_berlin --gate-mode standard "
            f"--seeds {seeds} --trials 10 --optimization-level 3 "
            f"--max-twoq-error 0.03 "
            f"--out results/nighthawk_standard_untwirled/{prefix}_preflight"
        )


if __name__ == "__main__":
    main()
