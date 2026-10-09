TARI.Miner C29 v1.1.7
=====================

Community Tari Cuckaroo29 CUDA miner. No developer fee.

Windows quick start
-------------------
1. Run start-c29.bat.
2. Paste your Tari wallet address when prompted.
3. Leave each GPU miner window open.

The starter is preconfigured for:
  Pool: taric29-ca.luckypool.io:3111
  Worker: your Windows computer name, plus -gpu0, -gpu1, and so on
  GPUs: all supported NVIDIA cards

The correct backend is selected separately for every RTX 30, RTX 40, and RTX
50 series GPU, including mixed-card rigs. The starter does not change clocks,
voltage, fans, or power limits.

To save settings, edit the optional TARI_WALLET, TARI_WORKER, or TARI_DEVICES
lines near the top of start-c29.bat. Use TARI_DEVICES=0,2 to mine only on
those device indexes.

Extra options
-------------
start-c29.bat --pipeline 1

Options after start-c29.bat are applied to every selected GPU. If automatic
VRAM allocation fails, use --pipeline 1.

The standalone solver backends are in bin. A correct solver test ends with
"verify failures: 0".
