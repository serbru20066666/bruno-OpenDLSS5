"""Diffusion backend: regenerate each frame with SD-Turbo image-to-image (ONNX Runtime + DirectML, any GPU).

One UNet evaluation per frame. The frame is encoded to a latent, noised to one of the timesteps the distilled model
was trained on, and the model predicts the clean latent directly:

    z_t = sqrt(a_t) * z + sqrt(1 - a_t) * noise          x0 = (z_t - sqrt(1 - a_t) * eps) / sqrt(a_t)

`strength` picks the timestep: more noise = more of the image is regenerated (skin, materials, lighting), less of the
original survives. The noise tensor is the same every frame, which removes most of the frame-to-frame flicker that
fresh noise would cause; it does not make the result temporally stable.

Needs the files written by scripts/export_sdturbo.py, plus torch + transformers for the text encoder (run once).
"""
import json
from pathlib import Path

import cv2
import numpy as np

TIMESTEPS = {1: 249, 2: 499, 3: 749, 4: 999}   # the four timesteps of the distilled (ADD) student


class SdTurboBackend:
    def __init__(self, model_dir, width, height, prompt, strength=2, seed=7):
        import onnxruntime as ort
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        model_dir = Path(model_dir)
        self.w, self.h = width, height
        # 1..4 = the student's own timesteps; a fraction 0..1 = any timestep (lower keeps more of the frame)
        self.t = TIMESTEPS[strength] if strength in TIMESTEPS else int(round(float(strength) * 999))
        self.name = f"sd-turbo {width}x{height} t={self.t}"

        sched = json.loads((model_dir / "scheduler" / "scheduler_config.json").read_text())
        betas = np.linspace(sched["beta_start"] ** 0.5, sched["beta_end"] ** 0.5, sched["num_train_timesteps"], dtype=np.float64) ** 2
        a = np.cumprod(1.0 - betas)[self.t]
        self.sa, self.s1a = np.float32(np.sqrt(a)), np.float32(np.sqrt(1.0 - a))
        self.v_prediction = sched.get("prediction_type", "epsilon") == "v_prediction"

        tok = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        enc = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32).eval()
        ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.text = enc(ids)[0].numpy().astype(np.float32)
        del enc

        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        onnx_dir = model_dir / "onnx"
        tag = f"{width}x{height}"
        self.enc = ort.InferenceSession(str(onnx_dir / f"vae_encoder_{tag}.onnx"), opts, providers=prov)
        self.unet = ort.InferenceSession(str(onnx_dir / f"unet_{tag}.onnx"), opts, providers=prov)
        self.dec = ort.InferenceSession(str(onnx_dir / f"vae_decoder_{tag}.onnx"), opts, providers=prov)
        quant = np.load(onnx_dir / "vae_quant.npz")   # 1x1 quant_conv, first 4 channels = the latent mean
        self.qw = (quant["weight"][:4] * quant["scaling"]).astype(np.float32)
        self.qb = (quant["bias"][:4] * quant["scaling"]).astype(np.float32).reshape(1, 4, 1, 1)
        self.noise = np.random.default_rng(seed).standard_normal((1, 4, height // 8, width // 8)).astype(np.float32)
        self.timestep = np.array([self.t], np.float32)
        print("sd-turbo providers:", self.unet.get_providers(), "| prompt:", prompt, flush=True)

    def __call__(self, bgr):
        src = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = (src[:, :, ::-1].astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        moments = self.enc.run(None, {"image": np.ascontiguousarray(x)})[0]
        z = np.einsum("oc,nchw->nohw", self.qw, moments) + self.qb
        zt = self.sa * z + self.s1a * self.noise
        pred = self.unet.run(None, {"sample": zt, "timestep": self.timestep, "text": self.text})[0]
        z0 = self.sa * zt - self.s1a * pred if self.v_prediction else (zt - self.s1a * pred) / self.sa
        img = self.dec.run(None, {"latent": z0.astype(np.float32)})[0][0]
        out = np.clip((img.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out[:, :, ::-1])
