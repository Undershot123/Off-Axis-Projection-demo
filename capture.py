# capture.py — Win32 window capture and enumeration.

import threading
import time

import numpy as np
import win32gui

import config


def _enum_windows_callback(hwnd, result_list):
    if not win32gui.IsWindowVisible(hwnd):
        return
    title = win32gui.GetWindowText(hwnd)
    if not title:
        return
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        if (r - l) < 10 or (b - t) < 10:
            return
    except Exception:
        return
    if title == "3D Full Grid Parallax":
        return
    result_list.append((hwnd, title))


def enumerate_windows():
    results = []
    win32gui.EnumWindows(_enum_windows_callback, results)
    return results


class WindowCapture:
    """
    Captures a Win32 window by HWND at ~CAPTURE_FPS using mss.
    Thread-safe: background thread writes under a lock; get_frame() returns a copy.
    """
    def __init__(self, hwnd, target_fps=None):
        self.hwnd        = hwnd
        self._target_fps = target_fps if target_fps is not None else config.CAPTURE_FPS
        self._frame      = None
        self._lock       = threading.Lock()
        self.running     = True
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import mss
        with mss.mss() as sct:
            while self.running:
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                    w = right - left
                    h = bottom - top
                    if w > 0 and h > 0:
                        monitor = {"left": left, "top": top, "width": w, "height": h}
                        img = sct.grab(monitor)
                        frame_bgr = np.array(img)[:, :, :3]
                        with self._lock:
                            self._frame = frame_bgr
                except Exception:
                    pass
                time.sleep(1 / self._target_fps)

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self.running = False
