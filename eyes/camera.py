"""Optional external camera eye. OpenCV is optional."""
from __future__ import annotations
try:
    import cv2
except Exception:
    cv2=None
class CameraEye:
    def __init__(self,index=0): self.index=index; self.cap=None
    def open(self):
        if cv2 is None: raise RuntimeError("Install opencv-python to use a camera eye")
        if self.cap is None:self.cap=cv2.VideoCapture(self.index)
        return bool(self.cap.isOpened())
    def read(self):
        if not self.open(): raise RuntimeError("Camera could not be opened")
        ok,frame=self.cap.read()
        if not ok: raise RuntimeError("Camera frame could not be read")
        return frame
    def close(self):
        if self.cap is not None:self.cap.release();self.cap=None
