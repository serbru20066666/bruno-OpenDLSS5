"""Rough cost model of the NR network on a GPU without matrix units (e.g. RDNA2).

1. Counts the multiply-accumulates per frame from the block schedule in src/nr_graph.cpp.
2. Measures achieved FP16 GEMM throughput on this machine through ONNX Runtime + DirectML.
3. Estimates ms/frame = MACs*2 / achieved throughput. Optimistic: ignores attention softmax, FP8 re-quantisation,
   transitions and memory traffic. Use it as a lower bound that tells whether a real RDNA2 backend is worth doing.

Usage: python rdna2_estimate.py [width height]
"""
import sys, time
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1280, 720)


def halve(v):
    return (((v + 1) // 2) + 3) // 4 * 4


def align(v, a):
    return (v + a - 1) // a * a


fw, fh = align(W, 64), align(H, 64)   # approximation of Geometry::fromValid's padded field
lv = []
w, h = fw, fh
for _ in range(6):
    w, h = halve(w), halve(h)
    lv.append(w * h)
T0 = fw * fh

def macs_window(C, T):
    if C == 32:
        ffn = 32 * 128 + 128 * 32
    elif C <= 256:
        ffn = (C // 32) * (C * 128 + 128 * 32) + C * C
    else:  # 512
        ffn = 8 * (64 * 256 + 256 * 64) + 512 * 512
    return T * (ffn + 3 * C * C + C * C + C * 128)   # + qkv + proj + 8x8 window attention (QK^T and PV)

def macs_vit(T):
    C = 1024
    return T * (2 * C * 4096 + 3 * C * C + C * C) + 2 * T * T * C

# (channels, blocks, tokens)  block 0 / 70 on the field, 1-4 / 66-69 on level 0, ...
sched = [(32, 2, T0), (32, 8, lv[0]), (64, 8, lv[1]), (128, 12, lv[2]), (256, 16, lv[3]), (512, 16, lv[4])]
macs = sum(macs_window(c, t) * n for c, n, t in sched) + 8 * macs_vit(lv[5])
print(f"{W}x{H} -> field {fw}x{fh}, tokens/level {[T0] + lv}")
print(f"MACs/frame: {macs/1e9:.1f} G  ({2*macs/1e12:.2f} TFLOP/frame)")

def bench_matmul(M, K, N, iters=20):
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT16, [M, K])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT16, [K, N])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [M, N])
    g = helper.make_graph([helper.make_node("MatMul", ["a", "b"], ["y"])], "g", [a, b], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    so = ort.SessionOptions(); so.enable_mem_pattern = False
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["DmlExecutionProvider"])
    A = (np.random.rand(M, K) * 0.1).astype(np.float16)
    B = (np.random.rand(K, N) * 0.1).astype(np.float16)
    for _ in range(3):
        s.run(None, {"a": A, "b": B})
    t = time.perf_counter()
    for _ in range(iters):
        s.run(None, {"a": A, "b": B})
    dt = (time.perf_counter() - t) / iters
    return 2 * M * K * N / dt / 1e12, dt * 1000

best = 0
for (M, K, N) in [(65536, 128, 128), (65536, 256, 256), (16384, 512, 512), (4096, 1024, 1024), (16384, 1024, 4096)]:
    tf, ms = bench_matmul(M, K, N)
    best = max(best, tf)
    print(f"FP16 GEMM {M}x{K}x{N}: {tf:.2f} TFLOPS ({ms:.2f} ms incl. host copies)")
print(f"best measured: {best:.2f} TFLOPS")
for eff in (1.0, 0.5, 0.25):
    ms = 2 * macs / (best * 1e12 * eff) * 1000
    print(f"  at {int(eff*100)}% of that: {ms:.0f} ms/frame  ({1000/ms:.1f} fps)")
