"""Export SD-Turbo (stabilityai/sd-turbo) to three fixed-size ONNX files for the diffusion backend of the frontend.

The weights are not part of this repository. Download the fp16 variants into models/sd-turbo first, e.g.

    from huggingface_hub import snapshot_download
    snapshot_download("stabilityai/sd-turbo", local_dir="models/sd-turbo",
                      allow_patterns=["*.json", "*.txt", "unet/*fp16.safetensors", "vae/*fp16.safetensors",
                                      "text_encoder/*fp16.safetensors"])

then

    python scripts/export_sdturbo.py --size 1024x576

Writes models/sd-turbo/onnx/{vae_encoder,unet,vae_decoder}_WxH.onnx in half precision with float32 inputs/outputs
(static shapes: they run faster on DirectML). The VAE encoder stays in float32 and stops before quant_conv,
whose weights go to vae_quant.npz (see VaeEncoder). The text encoder is not exported; the frontend runs it once per prompt
with PyTorch on the CPU. W and H must be multiples of 64.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from diffusers import AutoencoderKL, UNet2DConditionModel
from onnxconverter_common import float16


class VaeEncoder(torch.nn.Module):
    """The encoder up to the 8-channel moments, before quant_conv. DirectML refuses the graph when the 1x1 quant_conv
    and the slice to the 4 mean channels are part of it, so the frontend applies those itself (vae_quant.npz)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):   # [1, 3, H, W] in -1..1 -> [1, 8, H/8, W/8]
        return self.vae.encoder(image)


class VaeDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):  # scaled latent -> [1, 3, H, W] in -1..1
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample


class Unet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, text):
        return self.unet(sample, timestep, encoder_hidden_states=text, return_dict=False)[0]


def fix_types(model, exporter_casts):
    """Repair what the float16 converter leaves inconsistent when it runs without shape inference."""
    constants = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    for node in model.graph.node:
        if node.op_type == "Cast" and node.name in exporter_casts:
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = onnx.TensorProto.FLOAT16
        if node.op_type == "Resize":   # roi and scales must be float32 whatever the data type is
            for name in list(node.input)[1:3]:
                const = constants.get(name)
                for attr in const.attribute if const is not None else ():
                    if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT16:
                        attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float32), attr.t.name))


def export(module, inputs, names, path, half=True):
    tmp = path.parent / (path.stem + "_fp32")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, inputs, str(tmp / "model.onnx"), input_names=names, output_names=["out"], opset_version=17,
                      dynamo=False)
    model = onnx.load(str(tmp / "model.onnx"))   # pulls in the external data of models over 2 GB
    # The converter leaves the exporter's own Cast-to-float nodes (in the attention layers) at float32, which no longer
    # type-checks. Retarget exactly those; the casts the converter adds itself (around Resize, on the graph inputs and
    # outputs) must stay as they are.
    if not half:
        onnx.save(model, str(path))
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
        print("wrote", path, f"{path.stat().st_size / 1e6:.0f} MB (float32)", flush=True)
        return
    exporter_casts = {n.name for n in model.graph.node
                      if n.op_type == "Cast" and any(a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in n.attribute)}
    # shape inference inside the converter serialises the model, which fails above 2 GB (the fp32 UNet)
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    fix_types(model, exporter_casts)
    if model.ByteSize() > 1_500_000_000:   # protobuf cannot serialise 2 GB: keep the tensors in a side file
        onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=path.name + ".data")
    else:
        onnx.save(model, str(path))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print("wrote", path, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(root / "models" / "sd-turbo"))
    ap.add_argument("--size", default="1024x576")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    if w % 64 or h % 64:
        raise SystemExit("width and height must be multiples of 64")
    out = Path(a.model) / "onnx"
    out.mkdir(exist_ok=True)

    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", variant="fp16", torch_dtype=torch.float32).eval()
        export(VaeEncoder(vae), (torch.rand(1, 3, h, w) * 2 - 1,), ["image"], out / f"vae_encoder_{w}x{h}.onnx", half=False)
        export(VaeDecoder(vae), (torch.randn(1, 4, h // 8, w // 8),), ["latent"], out / f"vae_decoder_{w}x{h}.onnx")
        q = vae.quant_conv
        np.savez(out / "vae_quant.npz", weight=q.weight.detach().numpy()[:, :, 0, 0], bias=q.bias.detach().numpy(),
                 scaling=np.float32(vae.config.scaling_factor))
        del vae
        unet = UNet2DConditionModel.from_pretrained(a.model, subfolder="unet", variant="fp16", torch_dtype=torch.float32).eval()
        text_dim = unet.config.cross_attention_dim
        export(Unet(unet), (torch.randn(1, 4, h // 8, w // 8), torch.tensor([499.0]), torch.randn(1, 77, text_dim)),
               ["sample", "timestep", "text"], out / f"unet_{w}x{h}.onnx")


if __name__ == "__main__":
    main()
