"""Voice Agent: cleans the recording, finds the speech, and computes a 256-number speaker embedding."""
import io
import threading
from dataclasses import dataclass
from math import gcd

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class VoiceInfo:
    ok: bool = False
    reason: str = ""
    duration: float = 0.0
    speech_s: float = 0.0
    snr_db: float = 0.0
    clip_ratio: float = 0.0
    embedding: np.ndarray | None = None


class VoiceAgent:
    name = "VoiceAgent"
    SR = 16000
    MAX_SECONDS = 30

    def __init__(self, model_path, threads: int = 2):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=threads)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.lock = threading.Lock()

    def load(self, data: bytes):
        try:
            a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception:
            return None
        a = a.mean(axis=1)
        if sr != self.SR:
            g = gcd(sr, self.SR)
            a = resample_poly(a, self.SR // g, sr // g).astype(np.float32)
        return a

    def analyze(self, data: bytes) -> VoiceInfo:
        a = self.load(data)
        if a is None or len(a) < 1600:
            return VoiceInfo(reason="The audio could not be read")
        a = a[: self.SR * self.MAX_SECONDS]
        duration = len(a) / self.SR
        clip = float((np.abs(a) >= 0.99).mean())
        # Energy-based speech detection: 25 ms frames every 10 ms
        n = (len(a) - 400) // 160 + 1
        if n < 20:
            return VoiceInfo(duration=duration, reason="The recording is too short")
        idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
        rms = np.sqrt((a[idx] ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)
        noise = float(np.percentile(db, 10))
        mask = db > max(noise + 10, float(db.max()) - 40)
        if mask.sum() < 20:
            return VoiceInfo(duration=duration, clip_ratio=clip, reason="No speech detected")
        speech_s = float(mask.sum() * 0.01)
        snr = float(db[mask].mean() - noise)
        hop = 160 * np.arange(n)[:, None] + np.arange(160)[None, :]
        speech = a[hop][mask].reshape(-1)
        if speech_s < 1.0:
            return VoiceInfo(duration=duration, speech_s=speech_s, snr_db=snr, clip_ratio=clip,
                             reason="Not enough speech")
        with self.lock:
            s = self.ex.create_stream()
            s.accept_waveform(self.SR, speech.astype(np.float32))
            s.input_finished()
            emb = np.array(self.ex.compute(s), dtype=np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return VoiceInfo(ok=True, duration=duration, speech_s=speech_s, snr_db=snr, clip_ratio=clip, embedding=emb)
