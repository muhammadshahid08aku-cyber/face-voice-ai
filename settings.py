"""Settings: environment variables first, then Streamlit secrets, then defaults."""
import os
from dataclasses import dataclass


def get(name, default=None):
    v = os.getenv(name)
    if v not in (None, ""):
        return v
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


@dataclass
class Thresholds:
    # Cosine similarity thresholds. "accept" = confident match, "reject" = clearly different.
    # Scores in between are sent to human REVIEW. Calibrate on your own volunteers (Admin > Calibration).
    face_accept: float = 0.40
    face_reject: float = 0.25
    voice_accept: float = 0.65
    voice_reject: float = 0.45
    policy: str = "strict"          # strict = face AND voice must both match; balanced = weighted evidence
    w_face: float = 0.55            # balanced policy weight for the face (voice gets 1 - w_face)
    margin: float = 0.15            # identify mode: best candidate must beat the runner-up by this much
    # Quality gates
    min_face_px: int = 80
    min_sharpness: float = 20.0
    max_pose: float = 0.55
    min_speech_s: float = 2.0
    enroll_speech_s: float = 3.0
    min_snr_db: float = 8.0
    max_clip: float = 0.02
    # Enrollment
    min_face_samples: int = 3
    min_voice_samples: int = 3
    # Guardrails
    lock_failures: int = 5
    lock_minutes: int = 10
    max_image_mb: float = 8.0
    max_audio_mb: float = 10.0
    retention_days: int = 365


def load_thresholds() -> Thresholds:
    t = Thresholds()
    for f in t.__dataclass_fields__:
        v = get(f.upper())
        if v is None:
            continue
        cur = getattr(t, f)
        setattr(t, f, type(cur)(v))
    return t
