#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tari-c29-launcher-test.XXXXXX")"
SURVIVOR_PID_FILE="$TEMP_ROOT/survivor.pid"

cleanup() {
    if [[ -f "$SURVIVOR_PID_FILE" ]]; then
        survivor_pid="$(cat "$SURVIVOR_PID_FILE")"
        if [[ "$survivor_pid" =~ ^[0-9]+$ ]]; then
            kill -KILL "$survivor_pid" 2>/dev/null || true
        fi
    fi
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT

make_case() {
    local name="$1"
    local case_root="$TEMP_ROOT/$name"
    mkdir -p "$case_root/bin"
    cp "$ROOT/start-c29.sh" "$case_root/start-c29.sh"
    chmod +x "$case_root/start-c29.sh"
    printf '%s' "$case_root"
}

write_gpu_list() {
    local destination="$1"
    cat > "$destination" <<'EOF'
#!/usr/bin/env bash
printf '0, 12.0, Test GPU 0\n1, 8.9, Test GPU 1\n'
EOF
    chmod +x "$destination"
}

# Fake AMD backend: --list-devices prints DEVICES (printf format, one
# "index<TAB>name<TAB>arch<TAB>bus" line per GPU; older builds print no bus);
# a worker run just exits 0.
write_amd_miner() {
    local destination="$1"
    local devices="$2"
    cat > "$destination" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "--list-devices" ]]; then
    printf '$devices'
    exit 0
fi
echo "amd worker \$*"
exit 0
EOF
    chmod +x "$destination"
}

write_nvidia_miner() {
    local destination="$1"
    cat > "$destination" <<'EOF'
#!/usr/bin/env bash
echo "nvidia worker $*"
exit 0
EOF
    chmod +x "$destination"
}

# run_launcher CASE_ROOT [NAME=VALUE...]
run_launcher() {
    local case_root="$1"
    shift
    set +e
    env \
        TARI_WALLET=test-login \
        TARI_WORKER=rig \
        TARI_NVIDIA_SMI="$case_root/nvidia-smi" \
        SURVIVOR_PID_FILE="$SURVIVOR_PID_FILE" \
        "$@" \
        timeout 10s "$case_root/start-c29.sh" >"$case_root/output.log" 2>&1
    launcher_status=$?
    set -e
}

# expect_status CASE_ROOT STATUS LABEL
expect_status() {
    if ((launcher_status != $2)); then
        cat "$1/output.log"
        echo "$3: expected exit $2, got $launcher_status" >&2
        exit 1
    fi
}

# expect_output CASE_ROOT TEXT LABEL
expect_output() {
    if ! grep -qF -- "$2" "$1/output.log"; then
        cat "$1/output.log"
        echo "$3: expected output to contain [$2]" >&2
        exit 1
    fi
}

# expect_no_output CASE_ROOT TEXT LABEL
expect_no_output() {
    if grep -qF -- "$2" "$1/output.log"; then
        cat "$1/output.log"
        echo "$3: expected output not to contain [$2]" >&2
        exit 1
    fi
}

missing_root="$(make_case missing)"
write_gpu_list "$missing_root/nvidia-smi"
cat > "$missing_root/bin/tari_c29_pool_miner_sm_120" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$missing_root/bin/tari_c29_pool_miner_sm_120"
run_launcher "$missing_root"
if ((launcher_status != 5)); then
    cat "$missing_root/output.log"
    echo "expected missing backend exit 5, got $launcher_status" >&2
    exit 1
fi

wedged_root="$(make_case wedged)"
write_gpu_list "$wedged_root/nvidia-smi"
cat > "$wedged_root/bin/tari_c29_pool_miner_sm_120" <<'EOF'
#!/usr/bin/env bash
sleep 0.1
exit 5
EOF
cat > "$wedged_root/bin/tari_c29_pool_miner_sm_89" <<'EOF'
#!/usr/bin/env bash
trap '' TERM
printf '%s\n' "$$" > "$SURVIVOR_PID_FILE"
while :; do sleep 1; done
EOF
chmod +x \
    "$wedged_root/bin/tari_c29_pool_miner_sm_120" \
    "$wedged_root/bin/tari_c29_pool_miner_sm_89"
run_launcher "$wedged_root"
if ((launcher_status != 5)); then
    cat "$wedged_root/output.log"
    echo "expected worker exit 5, got $launcher_status" >&2
    exit 1
fi
survivor_pid="$(cat "$SURVIVOR_PID_FILE")"
if kill -0 "$survivor_pid" 2>/dev/null; then
    cat "$wedged_root/output.log"
    echo "launcher left wedged worker $survivor_pid running" >&2
    exit 1
fi
rm -f -- "$SURVIVOR_PID_FILE"

# AMD-only rig without nvidia-smi: NVIDIA is skipped with a notice and one
# worker starts per AMD GPU, named -amd<index>.
amd_only_root="$(make_case amd-only)"
# Device 1 reports a feature suffix; device 2 is an integrated GPU the backend
# has no code for, and must be skipped rather than started. Device 2 has no
# bus field, as an older backend prints it.
write_amd_miner "$amd_only_root/bin/tari_c29_pool_miner_amd" \
    '0\tAMD Radeon RX 7900 XTX\tgfx1100\t03\n1\tAMD Radeon RX 6800\tgfx1030:xnack-\t0a\n2\tAMD Radeon Graphics\tgfx1036\n'
run_launcher "$amd_only_root" TARI_DRY_RUN=1
expect_status "$amd_only_root" 0 "AMD-only rig"
expect_output "$amd_only_root" "NOTICE: nvidia-smi could not enumerate NVIDIA GPUs; starting AMD GPUs only." "AMD-only rig"
expect_output "$amd_only_root" "tari_c29_pool_miner_amd --device 0 " "AMD-only rig"
expect_output "$amd_only_root" "--worker rig-amd0" "AMD-only rig"
expect_output "$amd_only_root" "--worker rig-amd1" "AMD-only rig"
expect_output "$amd_only_root" "AMD GPU 1 [AMD Radeon RX 6800] gfx1030 -> amd, worker rig-amd1" "AMD-only rig"
expect_output "$amd_only_root" "WARNING: Skipping AMD GPU 2 [AMD Radeon Graphics]: gfx1036 is not supported." "AMD-only rig"
expect_no_output "$amd_only_root" "--worker rig-amd2" "AMD-only rig"
expect_output "$amd_only_root" "Started 2 miner worker(s) from 0 detected NVIDIA GPU(s) and 3 detected AMD GPU(s)." "AMD-only rig"

# Only an unsupported AMD GPU: nothing to start, as for an unsupported NVIDIA GPU.
igpu_root="$(make_case amd-igpu-only)"
write_amd_miner "$igpu_root/bin/tari_c29_pool_miner_amd" '0\tAMD Radeon 780M\tgfx1103\n'
run_launcher "$igpu_root" TARI_DRY_RUN=1
expect_status "$igpu_root" 4 "unsupported AMD GPU only"
expect_output "$igpu_root" "WARNING: Skipping AMD GPU 0 [AMD Radeon 780M]: gfx1103 is not supported." "unsupported AMD GPU only"

# The launcher's supported list must match the targets the build compiles.
launcher_arches="$(sed -n 's/^AMD_SUPPORTED_ARCHES="\(.*\)"$/\1/p' "$ROOT/start-c29.sh")"
for script in build_solver.sh build_pool_miner.sh; do
    build_arches="$(sed -n 's/^AMD_OFFLOAD_ARCHES="\${AMD_OFFLOAD_ARCHES:-\(.*\)}"$/\1/p' "$ROOT/$script")"
    if [[ -z "$launcher_arches" || "$launcher_arches" != "$build_arches" ]]; then
        echo "AMD_SUPPORTED_ARCHES [$launcher_arches] in start-c29.sh differs from $script [$build_arches]" >&2
        exit 1
    fi
done

# TARI_AMD_DEVICES selects AMD GPUs; TARI_DEVICES does not apply to them.
run_launcher "$amd_only_root" TARI_DRY_RUN=1 TARI_AMD_DEVICES=1 TARI_DEVICES=0
expect_status "$amd_only_root" 0 "TARI_AMD_DEVICES"
expect_output "$amd_only_root" "--worker rig-amd1" "TARI_AMD_DEVICES"
expect_no_output "$amd_only_root" "--worker rig-amd0" "TARI_AMD_DEVICES"
run_launcher "$amd_only_root" TARI_DRY_RUN=1 TARI_AMD_DEVICES=5
expect_status "$amd_only_root" 4 "TARI_AMD_DEVICES matching no GPU"

# HiveOS AMD rigs have an nvidia-smi that cannot reach a driver.
cat > "$amd_only_root/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
exit 9
EOF
chmod +x "$amd_only_root/nvidia-smi"
run_launcher "$amd_only_root" TARI_DRY_RUN=1
expect_status "$amd_only_root" 0 "AMD-only rig with a failing nvidia-smi"
expect_output "$amd_only_root" "--worker rig-amd0" "AMD-only rig with a failing nvidia-smi"

# The AMD backend path can be overridden.
mv "$amd_only_root/bin/tari_c29_pool_miner_amd" "$amd_only_root/amd-miner"
run_launcher "$amd_only_root" TARI_DRY_RUN=1 TARI_AMD_MINER="$amd_only_root/amd-miner"
expect_status "$amd_only_root" 0 "TARI_AMD_MINER"
expect_output "$amd_only_root" "amd-miner --device 0 " "TARI_AMD_MINER"

# Mixed rig: NVIDIA workers keep their -gpu<index> names and gpu<index>.log,
# AMD workers use -amd<index> and amd<index>.log.
mixed_root="$(make_case mixed)"
write_gpu_list "$mixed_root/nvidia-smi"
write_nvidia_miner "$mixed_root/bin/tari_c29_pool_miner_sm_120"
write_nvidia_miner "$mixed_root/bin/tari_c29_pool_miner_sm_89"
write_amd_miner "$mixed_root/bin/tari_c29_pool_miner_amd" '0\tAMD Radeon RX 9070 XT\tgfx1201\t05\n'
run_launcher "$mixed_root" TARI_DRY_RUN=1
expect_status "$mixed_root" 0 "mixed rig"
expect_output "$mixed_root" "--worker rig-gpu0" "mixed rig"
expect_output "$mixed_root" "--worker rig-gpu1" "mixed rig"
expect_output "$mixed_root" "--worker rig-amd0" "mixed rig"
expect_output "$mixed_root" "Started 3 miner worker(s) from 2 detected NVIDIA GPU(s) and 1 detected AMD GPU(s)." "mixed rig"
expect_no_output "$mixed_root" "NOTICE" "mixed rig"
run_launcher "$mixed_root" TARI_LOG_DIR="$mixed_root/logs"
expect_status "$mixed_root" 0 "mixed rig with logs"
for log_name in gpu0 gpu1 amd0; do
    if [[ ! -f "$mixed_root/logs/$log_name.log" ]]; then
        cat "$mixed_root/output.log"
        echo "mixed rig: missing $log_name.log" >&2
        exit 1
    fi
done
if ! grep -qF -- "amd worker --device 0 " "$mixed_root/logs/amd0.log"; then
    cat "$mixed_root/logs/amd0.log"
    echo "mixed rig: amd0.log does not hold the AMD worker output" >&2
    exit 1
fi

# The AMD backend is present but lists no devices: an NVIDIA rig runs as before.
no_amd_root="$(make_case no-amd-devices)"
write_gpu_list "$no_amd_root/nvidia-smi"
write_nvidia_miner "$no_amd_root/bin/tari_c29_pool_miner_sm_120"
write_nvidia_miner "$no_amd_root/bin/tari_c29_pool_miner_sm_89"
write_amd_miner "$no_amd_root/bin/tari_c29_pool_miner_amd" ''
run_launcher "$no_amd_root" TARI_DRY_RUN=1
expect_status "$no_amd_root" 0 "AMD backend with no devices"
expect_output "$no_amd_root" "Started 2 miner worker(s) from 2 detected NVIDIA GPU(s)." "AMD backend with no devices"
expect_no_output "$no_amd_root" "worker rig-amd" "AMD backend with no devices"

# ...and without nvidia-smi there are no GPUs at all, which is an error.
rm -f "$no_amd_root/nvidia-smi"
run_launcher "$no_amd_root" TARI_DRY_RUN=1
expect_status "$no_amd_root" 3 "no GPUs at all"
expect_output "$no_amd_root" "ERROR: nvidia-smi could not enumerate NVIDIA GPUs." "no GPUs at all"

# An AMD backend that cannot run (no ROCm runtime) counts as no AMD GPUs.
cat > "$no_amd_root/bin/tari_c29_pool_miner_amd" <<'EOF'
#!/usr/bin/env bash
echo "error while loading shared libraries: libamdhip64.so.7" >&2
exit 127
EOF
write_gpu_list "$no_amd_root/nvidia-smi"
run_launcher "$no_amd_root" TARI_DRY_RUN=1
expect_status "$no_amd_root" 0 "AMD backend that cannot run"
expect_output "$no_amd_root" "Started 2 miner worker(s) from 2 detected NVIDIA GPU(s)." "AMD backend that cannot run"

# nvidia-smi exits non-zero when it lost a GPU but still lists the others:
# start-c29.sh must still launch the healthy GPU and ignore the error line.
partial_root="$(make_case partial-nvidia)"
cat > "$partial_root/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '0, 8.9, Test GPU 0\n'
echo "Unable to determine the device handle for GPU0000:02:00.0: Unknown Error. It's lost."
exit 15
EOF
chmod +x "$partial_root/nvidia-smi"
write_nvidia_miner "$partial_root/bin/tari_c29_pool_miner_sm_89"
run_launcher "$partial_root" TARI_DRY_RUN=1
expect_status "$partial_root" 0 "partial nvidia-smi failure"
expect_output "$partial_root" "--worker rig-gpu0" "partial nvidia-smi failure"
expect_output "$partial_root" "Started 1 miner worker(s) from 1 detected NVIDIA GPU(s)." "partial nvidia-smi failure"

echo "shell launcher regressions passed"
