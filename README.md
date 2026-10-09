# TARI.Miner

TARI.Miner is an open-source NVIDIA CUDA miner for Tari (XTM) Cuckaroo29.
It connects to LuckyPool and has no developer fee, payout address, donation
schedule, or alternate mining connection.

## Download

Open the [latest release](https://github.com/tari-project/TARI.Miner/releases/latest)
and download one file for your operating system:

- `TARI.Miner-v1.1.7-windows.zip`
- `TARI.Miner-v1.1.7-linux.tar.gz`
- `tari-miner-hiveos-1.1.7.tar.gz` for HiveOS

Each package contains all supported GPU backends. The starter detects every
NVIDIA GPU and selects the correct backend for each card:

- Compute capability 8.6: RTX 30 series
- Compute capability 8.9: RTX 40 series
- Compute capability 12.0: RTX 50 series

Mixed rigs are supported. Each GPU runs in its own process with a worker name
such as `RIG01-gpu0` or `RIG01-gpu1`. Pipeline depth is also selected inside
each process from the free VRAM remaining after its first solver context starts.
Automatic selection can use up to five contexts on Linux and Windows TCC, and
up to four on Windows WDDM. An explicit `--pipeline` value from 1 through 5
overrides the automatic platform cap; allocation still stops safely if VRAM
runs out.

The starter does not change GPU clocks, voltage, fans, or power limits.

## Windows

1. Extract `TARI.Miner-v1.1.7-windows.zip`.
2. Run `start-c29.bat`.
3. Paste your Tari wallet address when prompted.
4. Leave the starter window open while mining.

`start-c29.bat` runs `start-c29.ps1`, which does the work. Both files must stay
in the same folder. Windows PowerShell 5.1 is included with Windows, so nothing
needs installing.

Every GPU worker runs inside the starter's own window rather than a window of
its own, and the starter stays open until the last worker has exited. Press
Ctrl+C there to stop them all at once: workers share the starter's console so
that a single interrupt reaches every one of them.

The BAT file is preconfigured for `taric29-ca.luckypool.io:3111` and uses the
Windows computer name as the base worker name. To save settings, edit the
optional variables near the top of `start-c29.bat`.

To mine only on selected GPUs, set a comma-separated device list:

```bat
set "TARI_DEVICES=0,2"
start-c29.bat
```

To write per-GPU output to files instead of the console, set a log directory:

```bat
set "TARI_LOG_DIR=%CD%\logs"
start-c29.bat
```

Each worker writes progress to `gpu0.log`, `gpu1.log`, and so on, including the
periodic speed report. Warnings and connection errors go to `gpu0.err.log`
alongside it. The two streams are separate files on Windows; the Linux starter
combines them into one.

In the speed report, `speed` is the graph rate over about the last 60
seconds and `avg` is the average since the miner started. The report keeps
coming every 15 seconds during the waits between reconnect attempts, so time
lost to a pool outage shows as a falling rate. It pauses while a connection
attempt itself is blocked (an unreachable pool host can block a connect for up
to about two minutes) and during the wait of up to 20 seconds for the first
job after connecting. HiveOS shows 0 once the newest report is more than 90
seconds old. `accepted` and `rejected` count shares since the miner started,
across reconnects, and `t` is the report time in Unix seconds. `stale`
counts graphs that were not searched for shares because the pool had already
moved to a higher block height. Pools that still accept shares for the
previous block for a short time may lose a few shares per block this way;
pass `--no-stale-skip` to search and submit that work anyway (`stale` then
stays 0).

Pools that expect `wallet/worker` rather than `wallet.worker` need the login
separator set alongside the pool:

```bat
set "TARI_POOL=POOL_HOST:PORT"
set "TARI_LOGIN_SEPARATOR=/"
start-c29.bat
```

Extra miner options are applied to every selected GPU:

```bat
start-c29.bat --pipeline 1
```

## Linux

Extract `TARI.Miner-v1.1.7-linux.tar.gz`, then run:

```bash
./start-c29.sh
```

The starter prompts for a wallet in an interactive terminal. Environment
variables can provide permanent settings:

```bash
TARI_POOL=taric29-ca.luckypool.io:3111 \
TARI_WORKER=RIG01 \
TARI_WALLET=YOUR_TARI_WALLET \
TARI_DEVICES=all \
TARI_LOGIN_SEPARATOR=. \
./start-c29.sh
```

Use `TARI_DEVICES=0,2` to select specific GPUs. Press Ctrl+C to stop all GPU
workers started by the script.

## HiveOS

Create a Custom miner in the Flight Sheet with these values:

```text
Miner name: tari-miner-hiveos
Installation URL: https://github.com/tari-project/TARI.Miner/releases/download/v1.1.7/tari-miner-hiveos-1.1.7.tar.gz
Hash algorithm: cuckaroo29
Wallet and worker template: %WAL%.%WORKER_NAME%
Pool URL: stratum+tcp://taric29-ca.luckypool.io:3111
Pass: x
```

The HiveOS launcher uses every supported GPU unless Extra config arguments
contains a selector such as `TARI_DEVICES=0,2`. Other extra arguments are
passed to each miner process, for example `--pipeline 1`. Per-GPU graph rates,
temperatures, fans, and share counters are reported to the HiveOS agent.

## Miner Options

Options placed after the starter command are passed to every selected GPU:

```text
--pool host:port        Pool endpoint
--wallet WALLET         Required Tari wallet address
--worker NAME           Worker name
--pass VALUE            Pool password; defaults to x
--login-separator S     Joins wallet and worker in the pool login; defaults to .
--intensity N           Duty cycle from 1 to 100 percent; defaults to 100
--pipeline N            Overlapped solver contexts; defaults automatically
--ntrims N              Even trim-round count; defaults to the build's (50 on sm_89)
--max-runtime-sec N     Stop after N seconds
--no-stale-skip         Also search and submit work for a superseded block
--version               Print version and exit
```

### Intensity

`--intensity` sets how much of the time the miner works. At 100, the default, it
never pauses. At 50 it idles for about as long as it works, roughly halving both
the graph rate and the load on the card. Lower values scale the same way, so 25
works about a quarter of the time. Useful for sharing a GPU with something else,
or for keeping a laptop cooler and quieter:

```bash
./start-c29.sh --intensity 50
```

```bat
start-c29.bat --intensity 50
```

This is not the same as `--pipeline`. Pipeline depth controls how many solver
contexts overlap, which is a memory and latency tuning knob, and lowering it to
fit VRAM does not reduce how hard the GPU is driven. Intensity inserts idle time
between graphs and is the setting to reach for when the goal is less load.

The starter supplies `--device`, `--pool`, `--wallet`, and `--worker` after
user options so each GPU always receives its detected device index and unique
worker name. If automatic allocation does not fit in VRAM, retry with
`--pipeline 1`.

### CPU-limited rigs

After the GPU trims a graph, each miner process searches the remaining edges
for cycles on one CPU thread. At the default 50 trim rounds that takes about
30 ms per graph on a fast desktop core, about 0.4 of a core per RTX 4080. On a
rig with a weak CPU and several GPUs (for example 2 cores feeding 4 or more
fast GPUs) the CPU, not the GPUs, becomes the limit: CPU usage sits near 100%
and each GPU's graph rate falls below what the same card makes in a desktop.

On such a rig add `--ntrims 60`. Ten more trim rounds cost about 0.8% of GPU
throughput (measured on an RTX 4080, sm_89) but cut the CPU work per graph by
about 30%. That is **estimated** to raise the graph rate a CPU-bound rig can
reach by about 40%; the estimate is modelled from RTX 4080 measurements and has
not yet been measured on a real weak host. The proofs found are the same.
Leave it off on a desktop or any rig with CPU to spare, where it only costs
the 0.8%. `--ntrims 56` is a milder step. Measurements and the estimate behind
this: `docs/ntrims_weak_host_2026-10-08.md`.

```bash
./start-c29.sh --ntrims 60
```

### Pool login format

The login sent to the pool is the wallet address, the separator, then the
worker name. LuckyPool expects `wallet.worker`, which is the default. Pools
expecting `wallet/worker` need the separator changed:

```bash
TARI_WALLET=YOUR_TARI_WALLET \
TARI_POOL=POOL_HOST:PORT \
TARI_LOGIN_SEPARATOR=/ \
./start-c29.sh
```

`TARI_LOGIN_SEPARATOR` sits alongside `TARI_POOL` because the two belong
together: a pool defines both the endpoint and the login format it accepts.
Passing `--login-separator` after the starter command works as well.

The wallet is checked before the first connection. Whitespace, control
characters, and inputs larger than the longest supported Tari text encoding are
rejected outright. A login with one of the two Base58 address lengths that uses
a `0`, `O`, `I`, or `l` emits a typo warning before mining starts. It is not
rejected solely for that warning, because pools exist that expect a username
rather than an address.

The pool connection is plain TCP. Do not use a sensitive password for
`--pass`; the default `x` is sufficient for LuckyPool.

### Exit Codes

A miner worker keeps running through anything it can recover from, including a
dropped connection and a pool outage. It exits non-zero only for a condition
that needs a restart or an operator, so a rig supervisor can act on the code:

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown, or `--max-runtime-sec` elapsed |
| 1 | Startup failure: sockets unavailable, no such CUDA device, or not enough VRAM for one solver |
| 2 | Invalid command line |
| 3 | A GPU solution failed host verification; the GPU or its tuning is suspect |
| 4 | The pool rejected the login repeatedly; check wallet, worker, password, and separator |
| 5 | Solver failure: a CUDA error, or three consecutive graphs with no surviving edges |
| 6 | The pool accepted the connection but never sent a job |
| 7 | The pool repeatedly sent invalid protocol data |

The starters propagate these. When a GPU worker exits non-zero, the starter
stops the remaining workers and exits with that same code, rather than carrying
on with its healthy GPUs — otherwise HiveOS never sees the failure. A starter
that fails before any worker runs uses its own codes: 2 for a missing wallet, 3
when `nvidia-smi` finds no GPU, 4 when `TARI_DEVICES` matches none, 5 for a
missing backend binary, and 130 for Ctrl+C.

## Test The Solver

The standalone solver checks GPU results with an independent CPU verifier.
Choose the backend matching the test GPU:

```bat
bin\tari_c29_solver_sm_89.exe --device 0 --count 320 --pipeline 2
```

```bash
bin/tari_c29_solver_sm_89 --device 0 --count 320 --pipeline 2
```

A correct run ends with `verify failures: 0`.

### Exact GPU recall regression

The GPU recall test compares exact `(header nonce, proof hash)` sets, not just
cycle counts. It independently packs every 42-edge proof, recalculates its
BLAKE2b-256 hash and difficulty, repeats both builds, and checks pipeline
parity. Build the shipped and conservative reference profiles for the GPU
architecture, then run:

```bat
build_solver.bat sm_120
build_solver.bat sm_120 reference
python tests\tari_c29_gpu_recall.py run --candidate bin\tari_c29_solver_sm_120.exe --reference bin\validation\tari_c29_solver_sm_120_reference.exe --arch sm_120 --output-dir validation --parity-pipeline 4
```

```bash
./build_solver.sh sm_89
./build_solver.sh sm_89 reference
python3 tests/tari_c29_gpu_recall.py run --candidate bin/tari_c29_solver_sm_89 --reference bin/validation/tari_c29_solver_sm_89_reference --arch sm_89 --output-dir validation --parity-pipeline 2
```

The default sequence tests 4,200 fixed graphs and saves the raw logs, JSONL
proof records, invoked commands, and comparison report. The reference binaries
remain under `bin/validation` and are not included in release packages.
The runner also rejects a binary whose embedded build target, runtime GPU
architecture, or compiled trim default does not match `--arch`. Throughput
measurements are a separate performance check and do not replace the exact
recall gate. Hosted CI compiles both profiles and exercises the verifier
fixtures; the full sequence runs only on a trusted host with the matching GPU.

## Build From Source

The repository includes its required third-party source under
`third_party/cuckoo`.

Windows requires Visual Studio C++ tools and NVIDIA CUDA Toolkit 13.2. From an
x64 Native Tools Command Prompt:

```bat
build.bat
build_all.bat
```

Linux requires a C++ toolchain and an NVIDIA CUDA toolkit that supports all
three target architectures:

```bash
./build_all.sh
```

`build_all` also compiles the non-release reference solvers used by the recall
test. Individual backends can be built with `build_solver` or
`build_pool_miner` and one of `sm_86`, `sm_89`, or `sm_120`; pass `reference`
as the second `build_solver` argument to build only that validation profile.

Release builds add the tuning flags listed in `build_flags/<arch>.flags` (one
nvcc flag per line; `#` starts a comment, blank lines are ignored). The same
file is used by the `.sh` and `.bat` scripts, and each build prints the
effective list. `sm_120` carries the tuned set, including 48 trim rounds;
`sm_86` and `sm_89` are empty until they are tuned on real hardware. An
architecture with no file builds with no extra flags. The reference profile
ignores these files and pins every option to reference behaviour.

To try a different set without editing files, for example during a tuning
sweep, set `TARI_ARCH_FLAGS`. A non-empty value replaces the file for that
build; a value of only a space builds with no extra flags:

```bash
TARI_ARCH_FLAGS="-DTARI_C29_DEFAULT_NTRIMS=48 -DROUND23_TPB=960" ./build_solver.sh sm_89
```

```bat
set "TARI_ARCH_FLAGS=-DTARI_C29_DEFAULT_NTRIMS=48 -DROUND23_TPB=960"
build_solver.bat sm_89
```

Experimental trim options, each off by default. Try them through
`TARI_ARCH_FLAGS`. Measured on an RTX 4080 (sm_89) so far: only
`SEEDA_CHECKPOINT=32` gained (+1.3%, now enabled in `build_flags/sm_89.flags`);
`LATE_ROUND_SELF_ZERO_IDX` (-0.4%), `LATE_ROUND_BPB` (no gain) and
`TRIM_CUDA_GRAPH` (**-17%**) did not. See `docs/sm89_results_2026-10-06.md`.

- `-DLATE_ROUND_SELF_ZERO_IDX=1`: rounds 2, 3 and the late rounds clear the
  bucket counts they read, replacing the per-round index memsets.
- `-DTRIM_CUDA_GRAPH=1`: each solver context gets its own CUDA stream, and
  Round 0 through the final edge count runs as one CUDA graph. It cost 17% on
  sm_89; don't enable it without a benchmark on the target arch.
- `-DLATE_ROUND_BPB=2`, `4` or `8`: each late-round block filters that many
  buckets with one shared bitmap, cutting the late-round grid by the same
  factor (default 1 keeps the existing kernel).
- `-DSEEDA_CHECKPOINT=8`, `16` or `32`: SeedA keeps the first C hashes of each
  64-edge block in registers and rehashes only from there, instead of
  rehashing the whole block (default 0 keeps `SEEDA_REHASH`).

`docs/spec2_gpu_validation.md` lists the GPU measurements and recall checks
that decide whether to enable them; `docs/seeda_checkpoint.md` has the spill
report and the steps for `SEEDA_CHECKPOINT`.

`tests/tari_c29_gpu_recall.py` takes the expected release trim count from
`-DTARI_C29_DEFAULT_NTRIMS=` in the same file (50 if absent). It honours
`TARI_ARCH_FLAGS` too, so keep it set to the value the candidate solver was
built with when running the recall test.

The standalone solver's summary also reports the surviving edges per graph,
the time of the host cycle search, and how busy that keeps the main thread
(`--recall-jsonl` records the same numbers). `tools/ntrims_sweep.ps1` and
`tools/ntrims_sweep.sh` use them to choose the trim count per architecture;
`docs/ntrims_sweep.md` has the steps.

## License

TARI.Miner is GPL-3.0-or-later. Required upstream licenses and attribution are
included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`third_party/cuckoo/LICENSE.txt`.

This is independent community software and is not affiliated with or endorsed
by Tari or LuckyPool.
