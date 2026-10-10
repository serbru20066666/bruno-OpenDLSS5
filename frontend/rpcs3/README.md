# RPCS3 frontend (prototype)

Captures the RPCS3 game window, processes each frame with a backend and shows a NR OFF / NR ON split view.

```
python -m venv .venv && .venv\Scripts\pip install onnxruntime-directml numpy opencv-python dxcam
.venv\Scripts\python rpcs3_nr.py --title "God of War III"                 # passthrough: capture/display overhead
.venv\Scripts\python rpcs3_nr.py --title "God of War III" --model m.onnx --infer-width 640
```

Move the mouse to move the divider; Q / Esc quits. Windows only (DXGI capture), primary monitor, run RPCS3
windowed or borderless. The native DLSS-NR Vulkan backend will plug in as another backend once the AMD port lands.
