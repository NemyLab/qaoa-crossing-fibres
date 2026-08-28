#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild only the graph-dependent HCP products. Patch extraction, TractSeg,
# TOM construction and patch selection are deliberately reused unchanged.
# This script performs no QPU submission.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BUILDER="${BUILDER:-${PROJECT_ROOT}/07_build_real_crossing_graph.py}"

SOURCE_REP_ROOT="${SOURCE_REP_ROOT:-${PROJECT_ROOT}/real_data/HCP_100307_R_89_89_80}"
SOURCE_30_ROOT="${SOURCE_30_ROOT:-${PROJECT_ROOT}/real_data_replication}"
OUT_REP_ROOT="${OUT_REP_ROOT:-${PROJECT_ROOT}/real_data_8n/HCP_100307_R_89_89_80}"
OUT_30_ROOT="${OUT_30_ROOT:-${PROJECT_ROOT}/real_data_8n_replication}"
VALIDATION_OUT="${VALIDATION_OUT:-${PROJECT_ROOT}/hcp_8n_graph_validation}"
MANIFEST="${MANIFEST:-${HERE}/hcp_30_patch_ids.txt}"
FORCE="${FORCE:-0}"

prepare_output_dir() {
    local target="$1"
    if [[ -e "${target}" ]]; then
        if [[ -d "${target}" ]] && \
           [[ -z "$(find "${target}" -mindepth 1 -print -quit)" ]]; then
            echo "Reusing empty output directory: ${target}"
            return
        fi
        if [[ "${FORCE}" != "1" ]]; then
            echo "ERROR: output already exists: ${target}" >&2
            echo "Set FORCE=1 to move it aside and rebuild." >&2
            exit 2
        fi
        local stamp
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        local backup="${target}.backup_${stamp}"
        mv "${target}" "${backup}"
        echo "Moved existing output to: ${backup}"
    fi
    mkdir -p "${target}"
}

resolve_patch_file() {
    local case_root="$1"
    local candidate
    for candidate in \
        "${case_root}/real_patch_arrays.npz" \
        "${case_root}/patch/real_patch_arrays.npz"
    do
        if [[ -s "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "ERROR: missing patch archive below ${case_root}" >&2
    echo "Tried: ${case_root}/real_patch_arrays.npz" >&2
    echo "       ${case_root}/patch/real_patch_arrays.npz" >&2
    return 1
}

build_one() {
    local patch_id="$1"
    local patch_file="$2"
    local graph_dir="$3"

    [[ -s "${patch_file}" ]] || {
        echo "ERROR: missing patch archive for ${patch_id}: ${patch_file}" >&2
        exit 3
    }
    mkdir -p "${graph_dir}"
    echo
    echo "Building ${patch_id}"
    echo "Patch archive: ${patch_file}"
    python "${BUILDER}" \
        --patch "${patch_file}" \
        --out "${graph_dir}" \
        --degree-cap 3 \
        --conflict-weight 1.0 \
        --anchor-weight 0.35 \
        --sigma-orientation-deg 18.0 \
        --sigma-displacement-deg 22.0 \
        --minimum-continuation-weight 0.01 \
        --anchor-mode boundary4 \
        --max-tom-anchor-angle 25.0
}

[[ -s "${BUILDER}" ]] || { echo "ERROR: missing builder: ${BUILDER}" >&2; exit 1; }
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing manifest: ${MANIFEST}" >&2; exit 1; }
[[ -s "${PROJECT_ROOT}/07_build_real_crossing_graph_base.py" ]] || {
    echo "ERROR: missing ${PROJECT_ROOT}/07_build_real_crossing_graph_base.py" >&2
    exit 1
}
grep -q '"voxel_neighbourhood": 8' \
    "${PROJECT_ROOT}/07_build_real_crossing_graph_base.py" || {
    echo "ERROR: the installed base builder is not the eight-neighbour version." >&2
    exit 1
}

mapfile_supported=0
if command -v mapfile >/dev/null 2>&1; then
    mapfile_supported=1
fi

patch_ids=()
if [[ "${mapfile_supported}" == "1" ]]; then
    mapfile -t patch_ids < <(grep -Ev '^[[:space:]]*(#|$)' "${MANIFEST}")
else
    while IFS= read -r patch_id; do
        [[ -n "${patch_id}" && "${patch_id}" != \#* ]] && patch_ids+=("${patch_id}")
    done < "${MANIFEST}"
fi

if [[ "${#patch_ids[@]}" -ne 30 ]]; then
    echo "ERROR: expected 30 patch identifiers; found ${#patch_ids[@]}" >&2
    exit 1
fi

# Resolve and validate every source archive before creating or moving outputs.
# The representative case and replication cases can legitimately use different
# directory layouts, so accept both <case>/real_patch_arrays.npz and
# <case>/patch/real_patch_arrays.npz.
representative_patch_file="$(resolve_patch_file "${SOURCE_REP_ROOT}")"
replication_patch_files=()
for patch_id in "${patch_ids[@]}"; do
    replication_patch_files+=(
        "$(resolve_patch_file "${SOURCE_30_ROOT}/${patch_id}")"
    )
done

echo "Resolved all 31 source patch archives."

prepare_output_dir "${OUT_REP_ROOT}"
prepare_output_dir "${OUT_30_ROOT}"
prepare_output_dir "${VALIDATION_OUT}"

build_one \
    "HCP_100307_R_89_89_80" \
    "${representative_patch_file}" \
    "${OUT_REP_ROOT}/graph_validation"

for index in "${!patch_ids[@]}"; do
    patch_id="${patch_ids[$index]}"
    build_one \
        "${patch_id}" \
        "${replication_patch_files[$index]}" \
        "${OUT_30_ROOT}/${patch_id}/graph_validation"
done

python "${HERE}/validate_hcp_8n_graphs.py" \
    --representative "${OUT_REP_ROOT}/graph_validation" \
    --replication-root "${OUT_30_ROOT}" \
    --manifest "${MANIFEST}" \
    --out "${VALIDATION_OUT}"

echo
echo "Eight-neighbour graph rebuild completed and validated."
echo "Representative: ${OUT_REP_ROOT}"
echo "Replication:    ${OUT_30_ROOT}"
echo "Validation:     ${VALIDATION_OUT}"
