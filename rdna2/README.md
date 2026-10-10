# RDNA2 backend (step 1)

`rdna2_net` records the whole NR graph (71 blocks, docs/network.md) into one Vulkan command buffer using plain
compute shaders with packed FP16 math, and times it with synthetic (constant) weights. It needs no matrix units, FP8
or NVIDIA extensions, only Vulkan 1.3 with `shaderFloat16` and `storageBuffer16BitAccess`.

```
powershell -File rdna2\build.ps1          # shaders + build\rdna2\rdna2_net.exe  (set $env:VCVARS64 if vcvars is not found)
build\rdna2\rdna2_net.exe 1280 720        # run from build\rdna2 (the .spv files are next to the exe)
```

Reference measurement, RX 6800 XT, 1280x720 (field 1280x768): about 68-75 ms per frame in steady state
(about 14 fps), with a per-stage breakdown printed. Runs are occasionally 2-3x slower when the GPU is shared with
other applications or changes clock state; compare the best runs.

Not done yet: numerical validation against the reference (this uses true exp instead of the bit trick, no FP8
rounding, constant weights so the output is not an image), kernel fusion, tuning of the slower stages (512ch blocks,
ViT), and loading real weights from a model directory.
