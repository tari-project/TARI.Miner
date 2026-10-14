TARI.Miner C29 v1.1.8 - HiveOS
================================

Community Tari Cuckaroo29 CUDA miner, with experimental AMD support. No
developer fee.

HiveOS custom miner setup
-------------------------
1. Create an XTM wallet entry containing your Tari wallet address.
2. Create a Flight Sheet and choose "Configure in miner" for the pool.
3. Select Custom miner and use these values:

   Miner name: tari-miner-hiveos
   Installation URL:
   https://github.com/tari-project/TARI.Miner/releases/download/v1.1.8/tari-miner-hiveos-1.1.8.tar.gz
   Hash algorithm: cuckaroo29
   Wallet and worker template: %WAL%.%WORKER_NAME%
   Pool URL: stratum+tcp://taric29-ca.luckypool.io:3111
   Pass: x

4. Apply the Flight Sheet. Open Miner Log to confirm each GPU is assigned an
   sm_86, sm_89, or sm_120 backend and that shares are accepted.

All supported NVIDIA GPUs are enabled by default. Mixed RTX 30, RTX 40, and
RTX 50 rigs are supported because the launcher selects a backend separately
for every GPU.

Optional extra config
---------------------
Use the Flight Sheet's Extra config arguments field for miner options:

  --pipeline 1

To select device indexes, add this token to Extra config arguments:

  TARI_DEVICES=0,2

Rigs with a weak CPU and several GPUs: each GPU's miner searches every graph
for cycles on one CPU thread, about 0.4 of a fast core per RTX 4080 at the
default 50 trim rounds. If the CPU sits near 100% while mining and per-GPU
graph rates are clearly below what the same cards make in a desktop, the CPU
is the limit. Then add this to Extra config arguments:

  --ntrims 60

It costs about 0.8% of GPU throughput (measured on an RTX 4080) but cuts the
CPU work per graph by about 30%, so a CPU-limited rig is estimated to mine
more (modelled from RTX 4080 measurements, not yet measured on a weak rig).
The proofs found are the same. Leave it off when the CPU has spare capacity.

The integration writes one log per GPU and reports per-GPU graph rates,
temperatures, fans, accepted shares, and rejected shares to the HiveOS agent.
The graph rate is averaged over about the last 60 seconds (earlier versions
showed the average since start), so it drops while the pool is unreachable.
A GPU whose log has had no speed report for 90 seconds shows 0.
It does not change clocks, voltage, fans, or power limits.

Because of this, the HiveOS low-hashrate watchdog sees a pool outage within
about 60-90 seconds. Prefer a miner restart, or a longer delay, over a rig
reboot in its settings, so a short pool outage does not reboot the rig.

Accepted and rejected shares are the pool's own replies to submitted shares.
They are not proof of payout; check the pool's dashboard for that.

AMD GPUs (experimental)
-----------------------
AMD support is experimental and NOT YET TESTED ON HARDWARE. Use it to help
test, not to mine unattended.

Supported AMD GPUs, all served by bin/tari_c29_pool_miner_amd:
  RDNA2: RX 6600/6650/6700/6750/6800/6900/6950 series (gfx1030-gfx1032)
  RDNA3: RX 7600/7700/7800/7900 series (gfx1100-gfx1102)
  RDNA4: RX 9060 XT and RX 9070 series (gfx1200, gfx1201)
Older AMD GPUs (Vega, Polaris) are not supported. Each solver context needs
about 6 GB of VRAM.

The ROCm HIP runtime must be installed: the AMD binaries link against
libamdhip64.so.7 and are built with ROCm 7.2.4, so use ROCm 7.2 or newer.
Check with:
  ldd bin/tari_c29_pool_miner_amd
  bin/tari_c29_pool_miner_amd --list-devices

On AMD and mixed rigs the launcher starts one worker per AMD GPU, named with
-amd0, -amd1, and so on. To select AMD GPUs, add this token to Extra config
arguments (TARI_DEVICES selects NVIDIA GPUs only):

  TARI_AMD_DEVICES=0,2

Unsupported AMD GPUs, such as Ryzen integrated graphics or an RX 6400/6500
(gfx1034), are skipped with a warning. If the HIP runtime itself has trouble
with one, hide it by adding a token such as this to Extra config arguments
(a comma-separated list of the device indexes to use; they are then
renumbered from 0):

  HIP_VISIBLE_DEVICES=0

AMD graph rates and share counters come from the miner logs. Temperatures and
fans come from the HiveOS agent's GPU stats, matched by PCI bus with the bus
the miner prints when it starts; without them they show as 0.

Please report AMD results, good or bad, as a GitHub issue with the GPU model,
the ROCm version, the --list-devices output, the first lines of the Miner Log
and its speed reports.

Manual reinstall on a rig
-------------------------
/hive/miners/custom/custom-get \
  https://github.com/tari-project/TARI.Miner/releases/download/v1.1.8/tari-miner-hiveos-1.1.8.tar.gz \
  -f

Then reapply the Flight Sheet or run `miner restart`.
