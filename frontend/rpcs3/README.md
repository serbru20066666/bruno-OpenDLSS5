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

Build `build/rdna2/rdna2.dll` first (`powershell -File rdna2/build.ps1`), then:

```
python rpcs3_nr.py --title "God of War" --backend rdna2 --gain 8          # live, NR OFF / NR ON split
python rpcs3_nr.py --backend rdna2 --gain 8 --image some.png              # one image -> some.png.nr.png
```

The captured frame is resized to `--size` (default 1280x720), run through the 71-block network and resized back, so the
emulator can stay at any resolution. About 65 ms of GPU time per frame on an RX 6800 XT (about 80 ms with the game
running).

**The weights are synthetic for now**, so the "NR ON" side shows the frame plus structured noise and a colour shift,
not a more realistic image; `--gain` scales the network residual to make it visible. It proves the path frame ->
network -> frame. A meaningful image needs real weights (a model directory loader is the next step).

## Regenerating the frame with diffusion (`--backend sdturbo`)

The backend that changes skin, materials and lighting: SD-Turbo image-to-image, one UNet evaluation per frame, through
ONNX Runtime + DirectML (works on AMD). The weights are open (stabilityai/sd-turbo, Stability AI Community License) and
are not part of this repository.

```
# once: fp16 weights into models/sd-turbo (see scripts/export_sdturbo.py), pip install torch diffusers transformers onnxconverter-common
python scripts/export_sdturbo.py --size 1024x576
python frontend/rpcs3/rpcs3_nr.py --title "God of War" --backend sdturbo --size 1024x576 --strength 0.25 --prompt "photorealistic photo, ..."
python frontend/rpcs3/rpcs3_nr.py --backend sdturbo --size 1024x576 --strength 0.25 --image frame.png
```

Measured on an RX 6800 XT at 1024x576: about 570 ms per frame (1.8 fps): VAE encoder 160 ms, UNet 210 ms, VAE decoder
160 ms.

What to expect:

* Close-ups work best: a face that fills the frame comes back with photographic skin, hair, cloth and lighting.
* Small subjects are damaged. The latent is 8x smaller than the frame, so a character 50 pixels wide is about 6 latent
  cells: faces, tattoos and armour details are reinvented. Wide gameplay shots of GoW3 show this on Kratos.
* `--strength` trades fidelity for regeneration. Above about 0.4 the model starts replacing the scene.
* The identity of a face changes, and nothing ties one frame to the next except a fixed noise tensor, so moving images
  flicker. This is not NVIDIA's neural rendering network and does not have its temporal stability.

## A real open AI model today (`--model`, ONNX + DirectML)

Until NR weights exist, a visible AI result comes from an open pretrained network: Real-ESRGAN's compact model
`realesr-general-x4v3` (BSD-3-Clause, github.com/xinntao/Real-ESRGAN). It is a 4x reconstruction network (detail and
edges), not neural rendering: it does not change materials, skin or lighting.

```
# once: download realesr-general-x4v3.pth (release v0.2.5.0) into models/realesrgan/, pip install torch (CPU is enough)
python scripts/convert_realesrgan.py --size 1280x720
python frontend/rpcs3/rpcs3_nr.py --title "God of War" --model models/realesrgan/realesr-general-x4v3_1280x720.onnx
```

Measured on an RX 6800 XT while the game was running: 640x360 input about 130 ms per frame (7.8 fps), 1280x720 input
about 390 ms (2.6 fps). It is only an improvement over the emulator rendering at the PS3's native 720p; with RPCS3
already rendering at 4K the native frame is sharper than the AI reconstruction.

## Capture

`--capture window` (default) copies the game window's own content with PrintWindow, so it works with other windows on
top, at about 50 ms per 4K grab. `--capture screen` uses desktop duplication: fast, but it captures whatever is on top
of the game, including the viewer. `--no-window --snapshot f.png --exit-after N` runs unattended and writes the split
view plus `f.png.off.png` / `f.png.on.png`.
