"""RPCS3 frontend for bruno-OpenDLSS5.

Captures the RPCS3 game window, runs it through a pluggable backend and shows a
NR OFF / NR ON split view (move the mouse to move the divider, Q/Esc quits).

Backends:
  passthrough   no processing (measures capture + display overhead)
  onnx          any image->image ONNX model via ONNX Runtime + DirectML (AMD/any GPU)

The native Vulkan DLSS-NR backend (src/) will plug in here once the AMD port exists.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time

import cv2
import numpy as np

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()


def find_window(title_part):
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_part.lower() in buf.value.lower():
                    found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found


def client_rect(hwnd):
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + r.right, pt.y + r.bottom


class Passthrough:
    name = "passthrough"

    def __call__(self, bgr):
        return bgr


class OnnxBackend:
    def __init__(self, path, providers=None):
        import onnxruntime as ort

        self.name = f"onnx:{path}"
        self.sess = ort.InferenceSession(
            path, providers=providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape   # [1, 3, H, W]; H / W are ints when the model has a fixed size
        self.fixed = (shape[3], shape[2]) if isinstance(shape[2], int) and isinstance(shape[3], int) else None
        print("providers:", self.sess.get_providers(), "| model input:", self.fixed or "dynamic", flush=True)

    def __call__(self, bgr):
        if self.fixed and (bgr.shape[1], bgr.shape[0]) != self.fixed:
            bgr = cv2.resize(bgr, self.fixed, interpolation=cv2.INTER_AREA)
        x = bgr[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.sess.run(None, {self.inp: x})[0][0]
        y = (np.clip(y, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        return np.ascontiguousarray(y)


class Rdna2Backend:
    """The NR network on the GPU through build/rdna2/rdna2.dll (Vulkan compute, no NVIDIA extensions).

    The frame is resized to the network size, run, and resized back. Weights are synthetic until a model directory
    loader exists, so the residual is not a meaningful image; `gain` scales it to make it visible.
    """

    def __init__(self, dll, width, height, gain):
        import ctypes as c
        import os

        dll = os.path.abspath(dll)
        self.lib = c.CDLL(dll)
        self.lib.nr_create.argtypes = [c.c_int, c.c_int, c.c_char_p]
        self.lib.nr_frame.argtypes = [c.c_void_p, c.c_void_p, c.c_float, c.c_float, c.c_float, c.c_float, c.c_float]
        self.lib.nr_frame.restype = c.c_double
        if self.lib.nr_create(width, height, os.path.dirname(dll).encode()) != 0:
            sys.exit("rdna2 backend failed to start")
        self.w, self.h, self.gain = width, height, gain
        self.out = np.empty((height, width, 4), np.uint8)
        self.gpu_ms = 0.0
        self.name = f"rdna2 {width}x{height} (synthetic weights)"

    def __call__(self, bgr):
        src = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        rgba = np.ascontiguousarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGBA))
        self.gpu_ms = self.lib.nr_frame(rgba.ctypes.data, self.out.ctypes.data, self.gain, 0.0, 0.0, 0.0, 0.0)
        return cv2.cvtColor(self.out, cv2.COLOR_RGBA2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["passthrough", "onnx", "rdna2", "sdturbo"], help="default: onnx if --model, else passthrough")
    ap.add_argument("--dll", default=str(__import__("pathlib").Path(__file__).resolve().parents[2] / "build" / "rdna2" / "rdna2.dll"))
    ap.add_argument("--size", default="1280x720", help="network resolution for the rdna2 / sdturbo backends (sdturbo: 1024x576)")
    ap.add_argument("--sd-model", default=str(__import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "sd-turbo"))
    ap.add_argument("--prompt", default="photorealistic photograph, realistic human skin with pores, real materials, "
                                        "cinematic lighting, sharp focus, highly detailed",
                    help="sdturbo: what the regenerated frame should look like; describe the scene for better fidelity")
    ap.add_argument("--strength", type=float, default=0.25,
                    help="sdturbo: 0..1, how much of the frame is regenerated (0.15 subtle, 0.25 default, 0.4 strong)")
    ap.add_argument("--gain", type=float, default=1.0, help="rdna2: scale of the network residual (debug aid)")
    ap.add_argument("--capture", choices=["window", "screen"], default="window",
                    help="window: the game window's own content, even when covered (PrintWindow, slower); "
                         "screen: desktop duplication of the window's area (fast, but captures whatever is on top)")
    ap.add_argument("--no-window", action="store_true", help="do not open the viewer (with --snapshot / --exit-after)")
    ap.add_argument("--exit-after", type=int, default=0, help="live mode: quit after this many frames (unattended tests)")
    ap.add_argument("--snapshot", help="live mode: write the split view to this file at frame 30 (and print timings)")
    ap.add_argument("--image", help="process one image file instead of capturing; writes <image>.nr.png and exits")
    ap.add_argument("--title", default="RPCS3", help="substring of the window title (e.g. 'God of War III')")
    ap.add_argument("--model", help="ONNX image->image model; omit for passthrough")
    ap.add_argument("--infer-width", type=int, default=0, help="downscale capture to this width before inference")
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()

    kind = a.backend or ("onnx" if a.model else "passthrough")
    if kind == "rdna2":
        w, h = (int(v) for v in a.size.lower().split("x"))
        backend = Rdna2Backend(a.dll, w, h, a.gain)
    elif kind == "sdturbo":
        from sdturbo import SdTurboBackend

        w, h = (int(v) for v in a.size.lower().split("x"))
        backend = SdTurboBackend(a.sd_model, w, h, a.prompt, a.strength)
    elif kind == "onnx":
        backend = OnnxBackend(a.model)
    else:
        backend = Passthrough()

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"cannot read {a.image}")
        out = backend(img)
        # compare at the backend's resolution, so the resize itself does not count as a difference
        base = img if out.shape[:2] == img.shape[:2] else cv2.resize(img, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
        diff = np.abs(out.astype(np.int16) - base.astype(np.int16))
        cv2.imwrite(a.image + ".nr.png", out)
        print(f"wrote {a.image}.nr.png | mean abs difference {diff.mean():.2f} / 255, max {diff.max()}", flush=True)
        return

    wins = find_window(a.title)
    if not wins:
        sys.exit(f"No visible window containing '{a.title}'. Start the game in RPCS3 first.")
    hwnd, title = wins[0]
    print("capturing:", title, flush=True)
    if a.capture == "screen":
        import dxcam

        cam = dxcam.create(output_idx=0, output_color="BGR")
    else:
        from wincap import WindowCapture

        wcap = WindowCapture(hwnd)
    split = [0.5]

    def on_mouse(ev, x, y, flags, param):
        if param["w"]:
            split[0] = min(max(x / param["w"], 0.0), 1.0)

    win = "bruno-OpenDLSS5  |  NR OFF <-> NR ON"
    mp = {"w": 0}
    if not a.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse, mp)

    def poll(ms=1):   # True = quit requested
        if a.no_window:
            return False
        return cv2.waitKey(ms) & 0xFF in (27, ord("q"))

    t_prev, ema_inf, ema_fps, shown = time.time(), 0.0, 0.0, 0
    while True:
        if a.capture == "screen":
            l, t, r, b = client_rect(hwnd)
            # clip to the monitor: a maximised or partly off-screen window reports a rectangle past its edges
            l, t, r, b = max(l, 0), max(t, 0), min(r, cam.width), min(b, cam.height)
            frame = cam.grab(region=(l, t, r, b)) if r - l >= 64 and b - t >= 64 else None
        else:
            frame = wcap.grab()
        if frame is None:
            if not user32.IsWindow(hwnd):
                print("the game window is gone; stopping", flush=True)
                break
            if poll(10):
                break
            time.sleep(0.05)
            continue
        src = frame
        if a.infer_width and src.shape[1] > a.infer_width:
            h = int(src.shape[0] * a.infer_width / src.shape[1]) // 8 * 8
            src = cv2.resize(src, (a.infer_width // 8 * 8, h), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        out = backend(src)
        ema_inf = 0.9 * ema_inf + 0.1 * (time.perf_counter() - t0) * 1000
        if out.shape[:2] != frame.shape[:2]:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            base = frame
        else:
            base = frame

        h, w = base.shape[:2]
        mp["w"] = w
        x = int(split[0] * w)
        view = base.copy()
        view[:, x:] = out[:, x:]
        cv2.line(view, (x, 0), (x, h), (255, 255, 255), 2)
        cv2.putText(view, "NR OFF", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(view, "NR ON", (w - 140, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        now = time.time()
        ema_fps = 0.9 * ema_fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(view, f"{backend.name} | infer {ema_inf:.1f} ms | {ema_fps:.0f} fps", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not a.no_window:
            cv2.imshow(win, view)
        shown += 1
        if a.snapshot and shown == min(30, a.exit_after or 30):
            cv2.imwrite(a.snapshot, view)
            cv2.imwrite(a.snapshot + ".off.png", base)
            cv2.imwrite(a.snapshot + ".on.png", out)
            gpu = getattr(backend, "gpu_ms", None)
            print(f"snapshot {a.snapshot} | capture {w}x{h} | backend {ema_inf:.1f} ms"
                  + (f" (GPU {gpu:.1f} ms)" if gpu else "") + f" | {ema_fps:.1f} fps", flush=True)
        if poll() or (a.exit_after and shown >= a.exit_after):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
