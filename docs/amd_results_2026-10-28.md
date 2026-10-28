# AMD results: Radeon AI PRO R9700, 2026-10-28

First run of the AMD backend on hardware, and the trail from "5x slower than
lolMiner" to the one access pattern that costs it. Numbers are from the v1.1.8
release and from PR #14 builds pulled off the CI artifact; nothing here was
built locally.

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700, gfx1201 (Navi 48, same silicon as the RX 9070 XT), 32 GB, PCI bus 03 |
| ROCm | 7.2.2 runtime on the host, binaries built with 7.2.4 in CI |
| Host | Linux 7.2.6, 16 CPU threads |
| Reference | RTX 4080 table in `docs/sm89_results_2026-10-06.md` |
| Comparison | lolMiner 1.98a, same pool and login, unthrottled: 12.4 g/s at about 200 W |

## Speed and correctness

`tari_c29_solver_amd --device 0`, 320 graphs at pipeline 1 and 2,000 graphs
at auto pipeline (it picks 5 contexts on 32 GB):

| build | 320 graphs, pipeline 1 | 2,000 graphs, auto |
|---|---|---|
| v1.1.8 release (`WARP_DST_ATOMICS=1`, emulated match) | 4.461 g/s | 4.234 g/s |
| PR #14 (`WARP_DST_ATOMICS=0` under HIP) | 3.782 g/s | 3.717, 4.213, 4.247 g/s |

`verify failures: 0` on every run. Gate 4a passed, and gate 4b matched all 110
proofs of the NVIDIA fixture for both the candidate and the reference build.
The pool miner ran 300 s against the stub pool with 25 of 25 submits
accepted, exit 0.

Samples 90 s into the 2,000-graph runs: 203 W, 189 W and 132 W as the junction
went from 88°C to 91°C and sclk throttled from 3180 to 2843 MHz, with mclk
pinned at 1258 MHz. The graph rate didn't move with the clock. From the
`gpu_metrics` sysfs blob during a pipeline 1 run: gfx activity 89%, memory
controller (UMC) activity 27 to 28%, 197 W. So it's neither ALU bound nor
DRAM bandwidth bound.

## Stage profile

`TRIM_STAGE_TIMING=1` build, `--count 200 --pipeline 1`, mean of graphs 11
to 200, ms per graph:

| stage | SeedA | SeedB | R0 | R1 | R2 | R3 | late | tail | total |
|---|---|---|---|---|---|---|---|---|---|
| R9700 | 23.8 | 18.2 | 71.5 | 28.1 | 5.9 | 6.6 | 48.1 | 0.17 | 202.4 |
| RTX 4080 | 12.5 | 14.3 | 18.7 | 7.1 | 3.7 | 2.5 | 13.3 | 0.22 | 72.2 |
| ratio | 1.9 | 1.3 | 3.8 | 4.0 | 1.6 | 2.6 | 3.6 | 0.8 | 2.8 |

SeedB stages its bucket writes through shared memory and lands near the raw
bandwidth ratio of the two cards. The Round kernels write straight to 16,384
buckets, and they are the gap.

## What didn't help

Existing options, one gfx1201 timing build each, 200 graphs at pipeline 1:

| variant | R0 | R1 | late | total | g/s |
|---|---|---|---|---|---|
| default | 71.5 | 28.2 | 48.1 | 202.4 | 4.314 |
| `ROUND0/1/23_TPB=256` | 77.9 | 23.5 | 48.0 | 203.0 | 4.309 |
| `ROUND0/1/23_TPB=512` | 72.4 | 23.1 | 48.1 | 198.7 | 4.388 |
| `LATE_ROUND_SELF_ZERO_IDX=1` | 71.5 | 28.1 | 48.0 | 202.9 | 4.314 |
| `LATE_ROUND_BPB=4` | 71.5 | 28.1 | 49.3 | 203.8 | 4.296 |
| `LATE_ROUND_BPB=4`, self zero, `ROUND_LATE_TPB=512` | 71.5 | 28.1 | 50.1 | 204.9 | 4.278 |
| `ROUND0_DST_HASH_DYNAMIC_BITS=12` and the sm_120 set | 99.0 | 27.4 | 48.1 | 229.5 | 3.869 |
| `ROUND_U64_EDGE_LOAD=1`, `*_COUNT_NORMAL_LOAD=1` | 71.5 | 26.4 | 47.2 | 200.0 | 4.369 |

Runtime `--trim-tpb` for the late rounds: 256 gives 44.4 ms, 320 (default)
48.1, 512 49.2, 1024 52.9.

Loads in flight per thread (`ROUND_MLP`, PR #14): 2, 4 and 8 all leave R0 at
71.5 ms, alone or with `TPB=512`. The ISA confirms the eight loads are issued
back to back, so R0 isn't waiting on load latency.

Dropping the emulated `__match_any_sync` (the v1.1.8 to PR #14 change above)
is neutral too. The loop is real, about 32 serialized `ds_bpermute` round
trips per bucket store, but it's hidden behind something slower.

## Diagnostics

`TARI_DIAG_ROUND` builds strip parts of Round's second pass. They give wrong
results and the solver stops them after three graphs, which is enough to time
Round 0:

| second pass | R0 ms |
|---|---|
| full (count, read again, atomic slot, scattered 8 B store) | 71.9 |
| counting pass only | 7.4 |
| atomics kept, store dropped | 10.1 |
| store kept, coalesced into the block's own region, no atomic | 20.4 |
| store scattered by bucket, no atomic (slot = lane index) | 94.9 |

Reading 4 GB twice and doing the shared-memory counting costs about 10 ms.
The scattered 8-byte store is the other 60.

## Microbenchmark

`tools/amd_scatter_bench.c` is an OpenCL program (the ROCm OpenCL runtime
compiles it, no hipcc needed) that times the same patterns in isolation on
128 M random edges:

| pattern | ms | M edges/ms |
|---|---|---|
| coalesced 8 B copy | 3.48 | 38.6 (617 GB/s) |
| 16,384 buckets, atomic slot, 8 B per lane (Round today) | 28.5 | 4.7 |
| 16,384 buckets, no atomic, 8 B per lane | 35.6 | 3.8 |
| atomic slot only, no store | 2.5 | 54 |
| 16,384 buckets, atomic, 32 B per lane | 9.6 | 14.0 |
| 128 buckets, atomic, 8 B per lane | 8.8 | 15.2 |
| 128 buckets, atomic, 32 B per lane | 3.8 | 35.4 (566 GB/s) |
| 1024-thread shared-memory 2-bit counting (Round pass 1) | 1.96 | 68 |

4.7 M edges/ms times Round 0's 324 M output edges is 69 ms, which is the
measured R0. RDNA does not combine 16,384 concurrent 8-byte write streams
the way Ada's 64 MB L2 does. 128 streams it combines fine, and 32-byte lanes
into 128 streams run at the card's bandwidth.

## What this means

No build flag fixes it. The Round kernels need to write the way SeedA and
SeedB already do on this hardware: a first level that scatters into 128
rows with 32-byte flushes staged in shared memory, then a second level
within the row. Two coalesced passes at 500+ GB/s cost far less than one
pass at 60 to 75 GB/s. By the microbenchmark, Round 0's write cost would go
from about 60 ms to under 10, and the late rounds, which have the same
structure, would follow. That puts the R9700 in the range of the 4080's
72 ms per graph, which is lolMiner's speed on this card.

Until then the AMD backend mines correctly at about a third of lolMiner's
rate, and `--intensity` only changes its duty cycle.
