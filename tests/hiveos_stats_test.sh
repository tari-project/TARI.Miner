#!/usr/bin/env bash
# Feeds pool miner speed lines through hiveos/h-stats.sh and checks that it
# reports the rolling rate and the accepted/rejected counters, and that pool
# text or stale lines cannot change them. Set AWK=mawk or AWK=gawk to run
# h-stats.sh with that awk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tari-hiveos-stats-test.XXXXXX")"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

mkdir -p "$TEMP_ROOT/bin" "$TEMP_ROOT/logs"
cat > "$TEMP_ROOT/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '0, 8.9, 00000000:01:00.0, 61, 45\n'
EOF
chmod +x "$TEMP_ROOT/bin/nvidia-smi"
if [[ -n "${AWK:-}" ]]; then
    awk_path="$(command -v "$AWK")" || { echo "awk not found: $AWK" >&2; exit 1; }
    ln -s "$awk_path" "$TEMP_ROOT/bin/awk"
    echo "using $awk_path"
fi
: > "$TEMP_ROOT/miner.env"

now="$(date +%s)"
fail=0

# Runs h-stats.sh against the log on stdin and checks the output contains
# each expected string.
check() {
    local name="$1"
    shift
    cat > "$TEMP_ROOT/logs/gpu0.log"
    local stats
    stats="$(
        PATH="$TEMP_ROOT/bin:$PATH" \
        CUSTOM_CONFIG_FILENAME="$TEMP_ROOT/miner.env" \
        TARI_LOG_DIR="$TEMP_ROOT/logs" \
        TARI_DEVICES=all \
            bash "$ROOT/hiveos/h-stats.sh"
    )"
    local expected
    for expected in "$@"; do
        if [[ "$stats" != *"$expected"* ]]; then
            echo "FAIL $name: expected $expected in $stats" >&2
            fail=1
        fi
    done
}

# The last report wins. Pool text after it, an old-format line, an over-long
# counter and the summary line must all be ignored.
check "latest report" '"hs":[13.650]' '"ar":[2,1]' '"temp":[61]' <<EOF
connecting to pool.example:3333 as wallet.worker
speed 9.00 g/s | avg 9.00 g/s | graphs=135 cycles=0 submitted=0 accepted=0 rejected=0 t=$((now - 30))
share diff=100 target=50 nonce=0000000000000001
speed 13.65 g/s | avg 12.10 g/s | graphs=7260 cycles=12 submitted=3 accepted=2 rejected=1 t=$now
pool error: {"id":999,"error":"speed 0 g/s accepted=0 rejected=0"}
share rejected: {"id":7,"error":"speed 0.00 g/s | avg 0.00 g/s | graphs=0 cycles=0 submitted=0 accepted=0 rejected=0 t=$now"}
speed 1.00 g/s | graphs=1 cycles=0 submitted=0 accepted=5 rejected=5
speed 99.00 g/s | avg 1.00 g/s | graphs=1 cycles=0 submitted=0 accepted=99999999999999999999 rejected=0 t=$now
speed 1e300 g/s | avg 1.00 g/s | graphs=1 cycles=0 submitted=0 accepted=0 rejected=0 t=$now

--- summary ---
graphs=7300 elapsed=600.00s speed=12.167 g/s cycles=12 submitted=3 verify_failures=0
EOF

# The miner's own output, shared with tests/tari_miner_stats_test.cpp, with
# the report time moved to now.
fixture="$(tr -d '\r' < "$ROOT/tests/fixtures/speed_line.txt" | sed "s/ t=[0-9]*/ t=$now/")"
check "miner fixture" '"hs":[13.650]' '"ar":[2,1]' <<EOF
$fixture
EOF

# A line from an older miner, without the trailing stale= field, still parses.
check "no stale field" '"hs":[13.650]' '"ar":[2,1]' <<EOF
${fixture% stale=*}
EOF

# Any other trailing field is rejected, such as diff= or xn= text from the
# tail of a split pool-text line.
check "unknown trailing fields" '"hs":[0.000]' '"ar":[0,0]' <<EOF
$fixture diff=5 xn=0123
EOF

# A report cut off mid-write (the log read while the line is being written)
# must not hide the fresh report before it.
check "half-written last line" '"hs":[13.650]' '"ar":[2,1]' < <(
    printf '%s\n%s' "$fixture" "${fixture% t=*} t=17"
)

# A report far in the future (clock stepped back, or forged) is not fresh.
check "future report" '"hs":[0.000]' '"ar":[2,1]' <<EOF
${fixture% t=*} t=$((now + 3600))
EOF

# Only the tail of a flooded log is read: junk before the report is skipped,
# and the latest report is still found under bounded junk after it.
{
    for ((i = 0; i < 20000; i++)); do
        echo "pool error: {\"id\":$i,\"error\":\"padding padding padding padding padding padding padding padding\"}"
    done
    echo "${fixture/13.65 g\/s/9.00 g\/s}"
    echo "$fixture"
    for ((i = 0; i < 5000; i++)); do
        echo "pool error: {\"id\":$i,\"error\":\"padding padding padding padding padding padding padding padding\"}"
    done
} > "$TEMP_ROOT/flood.log"
check "flooded log" '"hs":[13.650]' '"ar":[2,1]' < "$TEMP_ROOT/flood.log"

# h-stats.sh reads the last 1 MiB of the log.
window=1048576

# Prints exactly $1 bytes of junk lines (at least 1 byte).
pad() {
    local left="$1" line
    line="$(printf 'pool error: %087d' 0)"
    while ((left > 100)); do
        echo "$line"
        left=$((left - 100))
    done
    printf '%*s\n' $((left - 1)) ''
}

# The 1 MiB cut lands inside a logged new-job line whose pool-supplied text
# holds a forged report. That partial line must be dropped. The real report
# is before the cut, so nothing is reported.
forged="speed 999.00 g/s | avg 1.00 g/s | graphs=1 cycles=0 submitted=0 accepted=5000 rejected=0 t=$now"
{
    echo "$fixture"
    pad 4000
    echo "new job height=1 id=abc diff=5 xn=$forged"
    pad $((window - ${#forged} - 1))
} > "$TEMP_ROOT/cut.log"
if [[ "$(tail -c "$window" "$TEMP_ROOT/cut.log" | head -c 9)" != "speed 999" ]]; then
    echo "FAIL cut log: the 1 MiB window does not start at the forged report" >&2
    fail=1
fi
check "cut inside pool text" '"hs":[0.000]' '"ar":[0,0]' < "$TEMP_ROOT/cut.log"

# A log of exactly 1 MiB is read whole, so a report on its first line counts.
{
    echo "$fixture"
    pad $((window - ${#fixture} - 1))
} > "$TEMP_ROOT/exact.log"
if [[ "$(wc -c < "$TEMP_ROOT/exact.log" | tr -d ' ')" != "$window" ]]; then
    echo "FAIL exact log is not 1 MiB" >&2
    fail=1
fi
check "log of exactly 1 MiB" '"hs":[13.650]' '"ar":[2,1]' < "$TEMP_ROOT/exact.log"

# A report older than 90 s means the miner stopped reporting; the counters
# are still the last known ones (even if the log file mtime is older than 180 s).
check "stale report" '"hs":[0.000]' '"ar":[2,1]' <<EOF
speed 13.65 g/s | avg 12.10 g/s | graphs=7260 cycles=12 submitted=3 accepted=2 rejected=1 t=$((now - 200))
pool error: {"id":999,"error":"still connected"}
EOF
touch -d "@$((now - 300))" "$TEMP_ROOT/logs/gpu0.log"
old_mtime_stats="$(
    PATH="$TEMP_ROOT/bin:$PATH" \
    CUSTOM_CONFIG_FILENAME="$TEMP_ROOT/miner.env" \
    TARI_LOG_DIR="$TEMP_ROOT/logs" \
    TARI_DEVICES=all \
        bash "$ROOT/hiveos/h-stats.sh"
)"
for expected in '"hs":[0.000]' '"ar":[2,1]'; do
    if [[ "$old_mtime_stats" != *"$expected"* ]]; then
        echo "FAIL old log mtime: expected $expected in $old_mtime_stats" >&2
        fail=1
    fi
done

# Only pool text: nothing is reported.
check "forged only" '"hs":[0.000]' '"ar":[0,0]' <<EOF
pool error: {"id":999,"error":"speed 50 g/s accepted=7 rejected=0"}
EOF

# AMD workers log to amd<index>.log. Their temperature, fan and bus come
# from the HiveOS GPU stats file, whose AMD entries are HIP devices 0, 1, ...
# check_stats NAME PATH_PREFIX LOG_DIR GPU_STATS_JSON EXPECTED...
check_stats() {
    local name="$1" path_prefix="$2" log_dir="$3" gpu_stats="$4"
    shift 4
    local stats
    stats="$(
        PATH="$path_prefix:$TEMP_ROOT/bin:$PATH" \
        CUSTOM_CONFIG_FILENAME="$TEMP_ROOT/miner.env" \
        TARI_LOG_DIR="$log_dir" \
        TARI_DEVICES=all \
        GPU_STATS_JSON="$gpu_stats" \
            bash "$ROOT/hiveos/h-stats.sh"
    )"
    local expected
    for expected in "$@"; do
        if [[ "$stats" != *"$expected"* ]]; then
            echo "FAIL $name: expected $expected in $stats" >&2
            fail=1
        fi
    done
}

amd_root="$TEMP_ROOT/amd"
mkdir -p "$amd_root/bin" "$amd_root/amd-only-logs" "$amd_root/hidden-logs" \
    "$amd_root/no-bus-logs" "$amd_root/mixed-logs"
# HiveOS AMD rigs have an nvidia-smi that cannot reach a driver.
cat > "$amd_root/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
exit 9
EOF
chmod +x "$amd_root/bin/nvidia-smi"
# A CPU entry first, then the two AMD GPUs out of bus order, one with a domain;
# values as strings, as HiveOS writes them. Matching is by bus, not position.
cat > "$amd_root/gpu-stats.json" <<'EOF'
{"busids":["00:00.0","04:00.0","0000:03:00.0"],"brand":["cpu","amd","amd"],"temp":["0","57","55"],"fan":["0","35","30"]}
EOF
# amd_log BUS: the AMD miner's first log line, which carries the PCI bus.
amd_log() {
    printf 'TARI.Miner C29 v0 on AMD Radeon RX 7900 XTX (24 GB, gfx1100, bus %s)\n' "$1"
}
amd_slow="${fixture/13.65 g\/s/9.00 g\/s}"
{ amd_log 03; printf '%s\n' "$fixture"; } > "$amd_root/amd-only-logs/amd0.log"
{ amd_log 04; printf '%s\n' "$amd_slow"; } > "$amd_root/amd-only-logs/amd1.log"
check_stats "AMD-only rig" "$amd_root/bin" "$amd_root/amd-only-logs" "$amd_root/gpu-stats.json" \
    '"hs":[13.650,9.000]' '"ar":[4,2]' '"temp":[55,57]' '"fan":[30,35]' '"bus_numbers":[3,4]'
check_stats "AMD-only rig without GPU stats" "$amd_root/bin" "$amd_root/amd-only-logs" "$amd_root/missing.json" \
    '"hs":[13.650,9.000]' '"ar":[4,2]' '"temp":[0,0]' '"fan":[0,0]' '"bus_numbers":[3,4]'

# HIP_VISIBLE_DEVICES=1: HIP device 0 is the second AMD card, on bus 04.
{ amd_log 04; printf '%s\n' "$amd_slow"; } > "$amd_root/hidden-logs/amd0.log"
check_stats "AMD GPU renumbered by HIP_VISIBLE_DEVICES" "$amd_root/bin" "$amd_root/hidden-logs" "$amd_root/gpu-stats.json" \
    '"hs":[9.000]' '"temp":[57]' '"fan":[35]' '"bus_numbers":[4]'

# No bus in the log yet: 0s, never the stats of whichever card is first.
printf '%s\n' "$fixture" > "$amd_root/no-bus-logs/amd0.log"
check_stats "AMD log without a bus" "$amd_root/bin" "$amd_root/no-bus-logs" "$amd_root/gpu-stats.json" \
    '"hs":[13.650]' '"temp":[0]' '"fan":[0]' '"bus_numbers":[0]'

# Mixed rig: NVIDIA GPU 0 (bus 01, from nvidia-smi) and AMD GPU 0 (bus 03).
# Both report bus_numbers the same way, as decimal bus numbers.
printf '%s\n' "$fixture" > "$amd_root/mixed-logs/gpu0.log"
{ amd_log 03; printf '%s\n' "$amd_slow"; } > "$amd_root/mixed-logs/amd0.log"
check_stats "mixed rig" "$TEMP_ROOT/bin" "$amd_root/mixed-logs" "$amd_root/gpu-stats.json" \
    '"hs":[13.650,9.000]' '"ar":[4,2]' '"temp":[61,55]' '"fan":[45,30]' '"bus_numbers":[1,3]'

# The AMD miner prints the bus in the format matched above.
if ! grep -qF 'printf("TARI.Miner C29 %s on %s (%.0f GB, %s, bus %02x)\n",' "$ROOT/tari_c29_pool_miner.cu"; then
    echo "FAIL the AMD startup line format in tari_c29_pool_miner.cu changed" >&2
    fail=1
fi

# nvidia-smi exits non-zero when it lost a GPU but still lists the others:
# those must still be reported, and its error line ignored.
partial_root="$TEMP_ROOT/partial"
mkdir -p "$partial_root/bin" "$partial_root/logs"
cat > "$partial_root/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '0, 8.9, 00000000:01:00.0, 61, 45\n'
echo "Unable to determine the device handle for GPU0000:02:00.0: Unknown Error. It's lost."
exit 15
EOF
chmod +x "$partial_root/bin/nvidia-smi"
printf '%s\n' "$fixture" > "$partial_root/logs/gpu0.log"
check_stats "nvidia-smi lost a GPU" "$partial_root/bin" "$partial_root/logs" "$amd_root/missing.json" \
    '"hs":[13.650]' '"ar":[2,1]' '"temp":[61]' '"fan":[45]' '"bus_numbers":[1]'

# h-stats.sh relies on pool text never starting a log line. The miner
# prints pool text only at these call sites, each after a fixed prefix. A new
# sanitize_for_terminal call must be checked the same way and added here.
miner="$ROOT/tari_c29_pool_miner.cu"
sites="$(grep -c 'sanitize_for_terminal(' "$miner" || true)"
if [[ "$sites" != 4 ]]; then
    echo "FAIL expected 4 pool text log sites in tari_c29_pool_miner.cu, found $sites" >&2
    fail=1
fi
for prefix in '"pool login rejected: %s\n"' '"share rejected: %s\n"' \
        '"pool error: %s\n"' '"new job height=%llu id=%s xn=%s diff=%llu\n"'; do
    if ! grep -qF "$prefix" "$miner"; then
        echo "FAIL pool text log format $prefix not found" >&2
        fail=1
    fi
done

if ((fail)); then
    exit 1
fi
echo "hiveos stats parse tests passed"
