"""Orchestrator: receives a request (enroll, verify, identify, erase) and runs the specialist agents in order.

   Request -> Guardrail -> Face + Voice agents -> Quality -> matching -> Fusion & Decision -> Audit -> Explainer
"""
import time
from dataclasses import dataclass, field

import numpy as np

import models
from agents.audit_agent import AuditAgent
from agents.consent_agent import CONSENT_VERSION, ConsentPrivacyAgent
from agents.explainer_agent import ExplainerAgent
from agents.face_agent import FaceAgent
from agents.fusion_agent import FusionDecisionAgent
from agents.guardrail_agent import GuardrailAgent
from agents.quality_agent import QualityAgent
from agents.trace import Trace
from agents.voice_agent import VoiceAgent


@dataclass
class Outcome:
    label: str                       # ACCEPT | REVIEW | REJECT | RETRY | LOCKED | OK | ERROR
    message: str = ""
    person: dict | None = None
    face_score: float | None = None
    voice_score: float | None = None
    combined: float | None = None
    problems: list = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)
    ranking: list = field(default_factory=list)


def top2_mean(templates, q):
    sims = np.sort(np.stack(templates) @ q)[::-1][:2]
    return float(sims.mean())


class Orchestrator:
    def __init__(self, store, th):
        self.th = th
        self.store = store
        self.face = FaceAgent(models.ensure("yunet"), models.ensure("sface"))
        self.voice = VoiceAgent(models.ensure("speaker"))
        self.quality = QualityAgent(th)
        self.fusion = FusionDecisionAgent(th)
        self.guard = GuardrailAgent(th, store)
        self.audit = AuditAgent(store)
        self.consent = ConsentPrivacyAgent(th)
        self.explainer = ExplainerAgent()

    # ------------------------------------------------------------------ analysis shared by verify/identify
    def _analyze(self, image, audio, trace, enrolling=False):
        fi = vi = None
        problems = []
        if image is not None:
            t0 = time.perf_counter()
            fi = self.face.analyze(image)
            pf = self.quality.check_face(fi)
            problems += [f"Face: {p}" for p in pf]
            trace.add("FaceAgent + QualityAgent", "ok" if not pf else "retry",
                      f"faces={fi.n_faces} width={fi.width}px sharp={fi.sharpness:.0f} pose={fi.pose:.2f}" if fi.ok else fi.reason, t0)
        if audio is not None:
            t0 = time.perf_counter()
            vi = self.voice.analyze(audio)
            pv = self.quality.check_voice(vi, enrolling)
            problems += [f"Voice: {p}" for p in pv]
            trace.add("VoiceAgent + QualityAgent", "ok" if not pv else "retry",
                      f"speech={vi.speech_s:.1f}s snr={vi.snr_db:.0f}dB" if vi.ok else vi.reason, t0)
        return fi, vi, problems

    # ------------------------------------------------------------------ enroll
    def enroll(self, name, consent, typed_name, faces: list, voices: list, actor="admin", allow_duplicate=False) -> Outcome:
        tr = Trace()
        t0 = time.perf_counter()
        errs = self.consent.validate(name, consent, typed_name)
        tr.add("ConsentPrivacyAgent", "ok" if not errs else "blocked", "; ".join(errs) or "consent recorded", t0)
        if errs:
            return Outcome("ERROR", "Consent is required before enrolling.", problems=errs, trace=tr)

        face_embs, voice_embs, problems = [], [], []
        for i, img in enumerate(faces):
            fi, _, p = self._analyze(img, None, tr)
            if p:
                problems += [f"Photo {i + 1}: {x}" for x in p]
            else:
                face_embs.append(fi.embedding)
        for i, au in enumerate(voices):
            _, vi, p = self._analyze(None, au, tr, enrolling=True)
            if p:
                problems += [f"Recording {i + 1}: {x}" for x in p]
            else:
                voice_embs.append(vi.embedding)
        if len(face_embs) < self.th.min_face_samples:
            problems.append(f"Need at least {self.th.min_face_samples} good photos ({len(face_embs)} usable)")
        if len(voice_embs) < self.th.min_voice_samples:
            problems.append(f"Need at least {self.th.min_voice_samples} good recordings ({len(voice_embs)} usable)")
        if problems:
            return Outcome("RETRY", "Some samples were not usable.", problems=problems, trace=tr)

        # All samples must belong to one person
        t0 = time.perf_counter()
        for label, embs, floor in (("photos", face_embs, self.th.face_reject), ("recordings", voice_embs, self.th.voice_reject)):
            sims = [float(a @ b) for i, a in enumerate(embs) for b in embs[i + 1:]]
            if sims and min(sims) < floor:
                tr.add("Fusion/Consistency", "blocked", f"{label} do not look like the same person", t0)
                return Outcome("REJECT", f"The {label} do not seem to come from the same person. Please retake them.",
                               problems=[f"lowest similarity between {label}: {min(sims):.2f}"], trace=tr)
        tr.add("Fusion/Consistency", "ok", "all samples agree", t0)

        # Duplicate check against people already enrolled
        t0 = time.perf_counter()
        if not allow_duplicate:
            ftem, vtem = self.store.templates("face"), self.store.templates("voice")
            cf = np.mean(face_embs, axis=0); cf /= np.linalg.norm(cf)
            cv = np.mean(voice_embs, axis=0); cv /= np.linalg.norm(cv)
            for pid, tl in ftem.items():
                if top2_mean(tl, cf) >= self.th.face_accept and pid in vtem and top2_mean(vtem[pid], cv) >= self.th.voice_accept:
                    p = self.store.person(person_id=pid)
                    tr.add("Fusion/Duplicate", "blocked", f"matches existing person {p['ext_id']}", t0)
                    return Outcome("REJECT", f"This person already looks enrolled as {p['ext_id']}.", trace=tr)
        tr.add("Fusion/Duplicate", "ok", "no existing match", t0)

        pid, ext = self.store.add_person(name, CONSENT_VERSION, self.consent.retention_until())
        for e in face_embs:
            self.store.add_template(pid, "face", e)
        for e in voice_embs:
            self.store.add_template(pid, "voice", e)
        self.audit.record(actor, "enroll", person_id=pid, detail=f"{len(face_embs)} face, {len(voice_embs)} voice templates")
        tr.add("AuditAgent", "ok", "enrollment logged")
        return Outcome("OK", f"Enrolled {name}. Their ID code is {ext}.", person=self.store.person(person_id=pid), trace=tr)

    # ------------------------------------------------------------------ verify (1:1)
    def verify(self, ext_id, image=None, audio=None, session=None) -> Outcome:
        tr = Trace()
        t0 = time.perf_counter()
        person = self.store.person(ext_id=(ext_id or "").strip().upper())
        if not person or person["status"] != "active":
            # Same answer for unknown and erased codes, and it counts as a failed attempt for this session
            self.audit.record("kiosk", "verify", decision="REJECT", detail="unknown id", session=session)
            return Outcome("REJECT", "ID code not recognised.", trace=tr)
        pid = person["id"]
        probs = self.guard.check_inputs(image, audio)
        locked = self.guard.locked(session, pid)
        tr.add("GuardrailAgent", "blocked" if (probs or locked) else "ok", "; ".join(probs) or ("locked" if locked else "inputs ok"), t0)
        if locked:
            return Outcome("LOCKED", self.explainer.explain("LOCKED", None, None, None, locked=True), trace=tr)
        if probs:
            return Outcome("ERROR", probs[0], problems=probs, trace=tr)

        fi, vi, problems = self._analyze(image, audio, tr)
        if problems:
            return Outcome("RETRY", self.explainer.explain("RETRY", None, None, None, problems), problems=problems, trace=tr)

        t0 = time.perf_counter()
        fs = vs = None
        if fi is not None:
            tl = self.store.templates("face").get(pid)
            fs = top2_mean(tl, fi.embedding) if tl else None
        if vi is not None:
            tl = self.store.templates("voice").get(pid)
            vs = top2_mean(tl, vi.embedding) if tl else None
        tr.add("Matcher", "ok", f"face={fs if fs is None else round(fs, 3)} voice={vs if vs is None else round(vs, 3)}", t0)

        t0 = time.perf_counter()
        d = self.fusion.decide(fs, vs)
        tr.add("FusionDecisionAgent", d.label.lower(), f"policy={self.th.policy} combined={d.combined}", t0)
        msg = self.explainer.explain(d.label, fs, vs, d, person_name=person["name"] if d.label == "ACCEPT" else None)
        self.audit.record("kiosk", "verify", person_id=pid, decision=d.label, face=fs, voice=vs, detail=f"combined={d.combined}", session=session)
        tr.add("AuditAgent", "ok", "decision logged")
        return Outcome(d.label, msg, person=person if d.label == "ACCEPT" else None, face_score=fs, voice_score=vs,
                       combined=d.combined, trace=tr)

    # ------------------------------------------------------------------ identify (1:N)
    def identify(self, image=None, audio=None, session=None) -> Outcome:
        tr = Trace()
        t0 = time.perf_counter()
        probs = self.guard.check_inputs(image, audio)
        locked = self.guard.locked(session)
        tr.add("GuardrailAgent", "blocked" if (probs or locked) else "ok", "; ".join(probs) or ("locked" if locked else "inputs ok"), t0)
        if locked:
            return Outcome("LOCKED", self.explainer.explain("LOCKED", None, None, None, locked=True), trace=tr)
        if probs:
            return Outcome("ERROR", probs[0], problems=probs, trace=tr)
        fi, vi, problems = self._analyze(image, audio, tr)
        if problems:
            return Outcome("RETRY", self.explainer.explain("RETRY", None, None, None, problems), problems=problems, trace=tr)

        t0 = time.perf_counter()
        ftem = self.store.templates("face") if fi is not None else {}
        vtem = self.store.templates("voice") if vi is not None else {}
        cands = []
        for pid in set(ftem) | set(vtem):
            fs = top2_mean(ftem[pid], fi.embedding) if pid in ftem else None
            vs = top2_mean(vtem[pid], vi.embedding) if pid in vtem else None
            d = self.fusion.decide(fs, vs)
            cands.append((d.combined, pid, fs, vs, d))
        cands.sort(key=lambda x: -x[0])
        tr.add("Matcher", "ok", f"compared with {len(cands)} enrolled people", t0)
        if not cands:
            self.audit.record("kiosk", "identify", decision="REJECT", detail="nobody enrolled", session=session)
            return Outcome("REJECT", "Nobody is enrolled yet.", trace=tr)

        t0 = time.perf_counter()
        comb, pid, fs, vs, d = cands[0]
        label = d.label
        note = ""
        if label == "ACCEPT" and len(cands) > 1 and comb - cands[1][0] < self.th.margin:
            label, note = "REVIEW", " The next-best candidate was too close, so a human must confirm."
        tr.add("FusionDecisionAgent", label.lower(), f"best combined={comb} margin={'n/a' if len(cands) < 2 else round(comb - cands[1][0], 3)}", t0)
        person = self.store.person(person_id=pid)
        msg = self.explainer.explain(label, fs, vs, d, person_name=person["name"] if label == "ACCEPT" else None) + note
        self.audit.record("kiosk", "identify", person_id=pid if label != "REJECT" else None, decision=label, face=fs, voice=vs,
                          detail=f"combined={comb}", session=session)
        tr.add("AuditAgent", "ok", "decision logged")
        ranking = [{"id": self.store.person(person_id=p)["ext_id"], "face": f, "voice": v, "combined": c} for c, p, f, v, _ in cands[:5]]
        return Outcome(label, msg, person=person if label == "ACCEPT" else None, face_score=fs, voice_score=vs,
                       combined=comb, trace=tr, ranking=ranking)

    # ------------------------------------------------------------------ privacy operations
    def erase(self, person_id, actor="admin"):
        self.store.erase(person_id)
        self.audit.record(actor, "erase", person_id=person_id, detail="templates deleted on request or consent withdrawal")

    def purge(self, actor="system") -> int:
        n = self.store.purge_expired()
        if n:
            self.audit.record(actor, "retention_purge", detail=f"{n} expired people erased")
        return n
