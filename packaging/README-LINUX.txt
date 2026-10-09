TARI.Miner C29 v1.1.7
=====================

Community Tari Cuckaroo29 CUDA miner. No developer fee.

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

Press Ctrl+C to stop all GPU workers. The standalone solver backends are in
bin. A correct solver test ends with "verify failures: 0".
