# RDNA2 backend (step 1)

`rdna2_net` records the whole NR graph (71 blocks, docs/network.md) into one Vulkan command buffer using plain
compute shaders with packed FP16 math, and times it with synthetic (constant) weights. It needs no matrix units, FP8
or NVIDIA extensions, only Vulkan 1.3 with `shaderFloat16` and `storageBuffer16BitAccess`.

```
powershell -File rdna2\build.ps1          # shaders + build\rdna2\rdna2_net.exe  (set $env:VCVARS64 if vcvars is not found)
build\rdna2\rdna2_net.exe 1280 720        # run from build\rdna2 (the .spv files are next to the exe)
```

`rdna2_net.exe W H [frames] [profile]`: the optional `profile` argument prints the cost per kind of dispatch.

Reference measurement, RX 6800 XT, 1280x720 (field 1280x768), random synthetic weights scaled so activations stay
finite through all 71 blocks: about 66-71 ms per frame (about 14 fps). The tool prints the per-stage minimum over
the frames, because other applications take GPU time slices at random points of a frame.

What the profile shows:

* The full-resolution and level-0 blocks (32 channels) are bound by memory traffic, not arithmetic: each GEMM writes
  and re-reads whole activation tensors (the 128-channel FFN hidden layer alone is 252 MB at full resolution).
  Fusing a block so that intermediates stay on chip, as the reference `fused_block32` does, is the main lever left.
* Window attention is the most expensive kind of dispatch (about 16 ms in total).
* GEMMs with few rows (ViT, 512 channels) are bound by parallelism; fewer rows per thread helped the ViT
  (14 ms -> 9 ms). Taller tiles for narrow outputs were measured slower and are not used.
* There are 426 dispatches with a barrier after each; small ones cost about 0.02-0.03 ms regardless of size.

Not done yet: numerical validation against the reference (no FP8 rounding, reduction orders are not the reference's;
the attention exponential does use the network's bit trick), block fusion, loading real weights from a model
directory, and the exact `Geometry::fromValid` padding (only sizes whose levels halve exactly are accepted).
