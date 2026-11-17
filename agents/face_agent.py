"""Face Agent: finds the face, measures image quality, and computes a 128-number face embedding."""
import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceInfo:
    ok: bool = False
    reason: str = ""
    n_faces: int = 0
    width: int = 0
    det_score: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    pose: float = 0.0              # 0 = frontal, larger = turned away
    embedding: np.ndarray | None = None
    box: tuple | None = None


class FaceAgent:
    name = "FaceAgent"
    MAXDIM = 960

    def __init__(self, det_model, rec_model):
        self.det = cv2.FaceDetectorYN.create(str(det_model), "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_model), "")
        self.lock = threading.Lock()

    @staticmethod
    def decode(data: bytes):
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def analyze(self, data: bytes) -> FaceInfo:
        img = self.decode(data)
        if img is None:
            return FaceInfo(reason="The image could not be read")
        h, w = img.shape[:2]
        s = self.MAXDIM / max(h, w)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        with self.lock:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
            if faces is None or len(faces) == 0:
                return FaceInfo(reason="No face found")
            faces = faces[np.argsort(-(faces[:, 2] * faces[:, 3]))]
            f = faces[0]
            x, y, fw, fh = [int(v) for v in f[:4]]
            crop = img[max(0, y):max(0, y) + max(1, fh), max(0, x):max(0, x) + max(1, fw)]
            if crop.size == 0:
                return FaceInfo(n_faces=len(faces), reason="Face is cut off by the image edge")
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(cv2.resize(gray, (112, 112)), cv2.CV_64F).var())
            re_x, le_x, nose_x = f[4], f[6], f[8]
            pose = float(abs(nose_x - (re_x + le_x) / 2) / (abs(le_x - re_x) + 1e-6))
            aligned = self.rec.alignCrop(img, f)
            feat = self.rec.feature(aligned).flatten().astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-9)
        return FaceInfo(ok=True, n_faces=len(faces), width=fw, det_score=float(f[14]), sharpness=sharp,
                        brightness=float(gray.mean()), pose=pose, embedding=feat, box=(x, y, fw, fh))
