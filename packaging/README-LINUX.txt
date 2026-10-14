TARI.Miner C29 v1.1.8
=====================

Community Tari Cuckaroo29 CUDA miner, with experimental AMD support. No
developer fee.

Linux quick start
-----------------
1. Open a terminal in this folder.
2. Start mining:
     ./start-c29.sh

The starter prompts for your wallet and is preconfigured for:
  Pool: taric29-ca.luckypool.io:3111
  Worker: your host name, plus -gpu0, -gpu1, and so on
  GPUs: all supported NVIDIA cards

The correct backend is selected separately for every RTX 30, RTX 40, and RTX
50 series GPU, including mixed-card rigs. The starter does not change clocks,
voltage, fans, or power limits.

Optional settings
-----------------
TARI_POOL=taric29-ca.luckypool.io:3111 \
TARI_WORKER=RIG01 \
TARI_WALLET=YOUR_TARI_WALLET \
TARI_DEVICES=0,2 \
./start-c29.sh

Options after start-c29.sh are applied to every selected GPU. If automatic
VRAM allocation fails, use --pipeline 1.

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

The starter finds AMD GPUs itself and starts one worker per GPU, named with
-amd0, -amd1, and so on, next to any NVIDIA workers. Select AMD GPUs with
TARI_AMD_DEVICES=0,2 (TARI_DEVICES selects NVIDIA GPUs only). A rig with only
AMD GPUs does not need nvidia-smi.
Unsupported AMD GPUs, such as Ryzen integrated graphics or an RX 6400/6500
(gfx1034), are skipped with a warning. If the HIP runtime itself has trouble
with one, hide it with HIP_VISIBLE_DEVICES=0 (a comma-separated list of the
device indexes to use; they are then renumbered from 0).

Please report AMD results, good or bad, as a GitHub issue with the GPU model,
the ROCm version, the --list-devices output, the first lines of the miner log
and its speed reports.

Press Ctrl+C to stop all GPU workers. The standalone solver backends are in
bin. A correct solver test ends with "verify failures: 0".
