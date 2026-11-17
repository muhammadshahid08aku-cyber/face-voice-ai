"""Model files: downloaded on first start, checked by SHA-256, cached on disk.

Face detector : YuNet  (OpenCV Zoo)
Face embedder : SFace  (OpenCV Zoo)
Speaker embedder: WeSpeaker ResNet34 (via sherpa-onnx release)
Check each model's licence and training-data terms before any commercial use.
"""
import hashlib
import urllib.request
from pathlib import Path

from settings import get

MODELS = {
    "yunet": dict(
        file="face_detection_yunet_2023mar.onnx",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": dict(
        file="face_recognition_sface_2021dec.onnx",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "speaker": dict(
        file="wespeaker_en_voxceleb_resnet34.onnx",
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
        sha256="5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94"),
}


def model_dir() -> Path:
    d = Path(get("MODEL_DIR", str(Path(__file__).parent / "model_cache")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(key: str) -> Path:
    spec = MODELS[key]
    path = model_dir() / spec["file"]
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    part = path.with_suffix(".part")
    with urllib.request.urlopen(spec["url"], timeout=180) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if _sha256(part) != spec["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model {spec['file']} failed its integrity check")
    part.replace(path)
    return path
