# RPCS3 frontend (prototype)

Captures the RPCS3 game window, processes each frame with a backend and shows a NR OFF / NR ON split view.

```
python -m venv .venv && .venv\Scripts\pip install onnxruntime-directml numpy opencv-python dxcam
.venv\Scripts\python rpcs3_nr.py --title "God of War III"                 # passthrough: capture/display overhead
.venv\Scripts\python rpcs3_nr.py --title "God of War III" --model m.onnx --infer-width 640
```

Move the mouse to move the divider; Q / Esc quits. Windows only (DXGI capture), primary monitor, run RPCS3
windowed or borderless. The native DLSS-NR Vulkan backend will plug in as another backend once the AMD port lands.

## The NR network on the GPU (`--backend rdna2`)

Build `builddna2dna2.dll` first (`powershell -File rdna2uild.ps1`), then:

```
python rpcs3_nr.py --title "God of War III" --backend rdna2 --gain 8      # live, NR OFF / NR ON split
python rpcs3_nr.py --backend rdna2 --gain 8 --image some.png              # one image -> some.png.nr.png
```

The captured frame is resized to `--size` (default 1280x720), run through the 71-block network and resized back, so the
emulator can stay at any resolution. About 65 ms of GPU time per frame on an RX 6800 XT.

**The weights are synthetic for now**, so the "NR ON" side shows the frame plus structured noise and a colour shift,
not a more realistic image; `--gain` scales the network residual to make it visible. It proves the path frame ->
network -> frame. A meaningful image needs real weights (a model directory loader is the next step).
