"""Fusion & Decision Agent: turns face and voice similarity scores into ACCEPT, REVIEW or REJECT.

Each score is rescaled so that 0 = the "clearly different" threshold and 1 = the "confident match"
threshold. The decision is made by fixed, auditable rules. No language model is involved.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decision:
    label: str                      # ACCEPT | REVIEW | REJECT
    combined: float = 0.0
    n_face: float | None = None
    n_voice: float | None = None
    reasons: list = field(default_factory=list)


class FusionDecisionAgent:
    name = "FusionDecisionAgent"

    def __init__(self, th):
        self.th = th

    @staticmethod
    def _norm(s, reject, accept):
        return float(np.clip((s - reject) / (accept - reject), 0.0, 1.25))

    def decide(self, face: float | None, voice: float | None) -> Decision:
        t = self.th
        nf = None if face is None else self._norm(face, t.face_reject, t.face_accept)
        nv = None if voice is None else self._norm(voice, t.voice_reject, t.voice_accept)
        reasons = []
        have = [x for x in (nf, nv) if x is not None]
        if not have:
            return Decision("REJECT", reasons=["No face or voice evidence was provided"])

        if t.policy == "strict":
            if nf is None or nv is None:
                reasons.append("Strict policy needs both the face and the voice")
                comb = min(min(have), 0.49)
            else:
                comb = min(nf, nv)
        else:  # balanced
            w = {"face": t.w_face, "voice": 1 - t.w_face}
            parts = [(nf, w["face"]), (nv, w["voice"])]
            parts = [(n, wt) for n, wt in parts if n is not None]
            comb = sum(n * wt for n, wt in parts) / sum(wt for _, wt in parts)
            if len(parts) < 2:
                reasons.append("Only one of face or voice was provided")
                comb = min(comb, 0.99)
            if min(have) < 0.25:
                reasons.append("One signal clearly disagrees with the claimed identity")
                comb = min(comb, 0.49)

        label = "ACCEPT" if comb >= 1.0 else "REVIEW" if comb >= 0.5 else "REJECT"
        return Decision(label, round(float(comb), 3), nf, nv, reasons)
