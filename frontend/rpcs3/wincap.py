"""Capture one window's own content with PrintWindow(PW_RENDERFULLCONTENT), even when other windows cover it.

Slower than desktop duplication (a GDI copy per frame) but it follows the window, not the screen, so the viewer or any
other window on top does not end up in the frame.
"""
import ctypes
import ctypes.wintypes as wt

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
PW_CLIENTONLY, PW_RENDERFULLCONTENT = 1, 2

user32.GetDC.restype = wt.HDC
user32.GetDC.argtypes = [wt.HWND]
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, wt.UINT]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wt.HDC]
gdi32.GetDIBits.argtypes = [wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT, ctypes.c_void_p, ctypes.c_void_p, wt.UINT]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


class WindowCapture:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.size = None
        self.screen = self.mem = self.bmp = None

    def _alloc(self, w, h):
        self.close()
        self.screen = user32.GetDC(None)
        self.mem = gdi32.CreateCompatibleDC(self.screen)
        self.bmp = gdi32.CreateCompatibleBitmap(self.screen, w, h)
        gdi32.SelectObject(self.mem, self.bmp)
        self.size = (w, h)
        self.buf = np.empty((h, w, 4), np.uint8)
        self.info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)

    def grab(self):
        """BGR frame of the client area, or None if the window has no area or the copy failed."""
        r = wt.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        w, h = r.right, r.bottom
        if w < 64 or h < 64:
            return None
        if self.size != (w, h):
            self._alloc(w, h)
        if not user32.PrintWindow(self.hwnd, self.mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            return None
        gdi32.GetDIBits(self.mem, self.bmp, 0, h, self.buf.ctypes.data, ctypes.byref(self.info), 0)
        return self.buf[:, :, :3]

    def close(self):
        if self.bmp:
            gdi32.DeleteObject(self.bmp)
            gdi32.DeleteDC(self.mem)
            user32.ReleaseDC(None, self.screen)
            self.bmp = None
