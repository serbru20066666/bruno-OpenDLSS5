"""Convert Real-ESRGAN's compact model (realesr-general-x4v3, BSD-3-Clause, github.com/xinntao/Real-ESRGAN) to ONNX.

The weights are not part of this repository: download realesr-general-x4v3.pth from the Real-ESRGAN release v0.2.5.0
into models/realesrgan/ and run

    python scripts/convert_realesrgan.py --size 640x360

The .pth is loaded with weights_only=True (tensors only, no code from the file is executed). The exported network takes
RGB float [1, 3, H, W] in 0..1 and returns [1, 3, 4H, 4W]; the size is fixed in the file because static shapes run
faster on DirectML.
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Same layout as basicsr/realesrgan's SRVGGNetCompact, so the published state dict loads as is."""

    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default=str(root / "models" / "realesrgan" / "realesr-general-x4v3.pth"))
    ap.add_argument("--size", default="640x360", help="input WxH; the output is 4x this")
    ap.add_argument("--fp16", action="store_true", help="store the network in half precision")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))

    state = torch.load(a.pth, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    net = SRVGGNetCompact()
    net.load_state_dict(state, strict=True)
    net.eval()

    out = Path(a.pth).with_name(f"realesr-general-x4v3_{w}x{h}{'_fp16' if a.fp16 else ''}.onnx")
    x = torch.rand(1, 3, h, w)
    torch.onnx.export(net, x, str(out), input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
    if a.fp16:
        import onnx
        from onnxconverter_common import float16

        onnx.save(float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True), str(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
