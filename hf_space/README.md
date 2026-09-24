---
title: zvuv brain
emoji: 🪰
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Full FlyWire fly brain (Shiu et al. 2024) for zvuv
---

# 🪰 zvuv: the full fly brain

A WebSocket server that runs the whole-brain spiking model of *Drosophila* from
Shiu et al. 2024 (Nature): 138,639 FlyWire v783 neurons and 15 million connections, in Brian2.

The [zvuv control site](https://itaydaniel1000-beep.github.io/zvuv/) connects to
`wss://<this space>.hf.space/`, sends what the virtual fly sees and smells,
and gets back firing rates. The site uses the full brain for the escape reflex
(LPLC2/LC4 → Giant Fiber) and its simple 16-node brain for odor and light steering.

Code: https://github.com/itaydaniel1000-beep/zvuv (the Dockerfile clones it at build time).
To update after a change on GitHub: Settings → Factory rebuild.
