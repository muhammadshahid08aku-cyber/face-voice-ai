"""Quality Agent: rejects poor photos and recordings before they can cause a wrong match."""


class QualityAgent:
    name = "QualityAgent"

    def __init__(self, th):
        self.th = th

    def check_face(self, fi) -> list[str]:
        t = self.th
        if not fi.ok:
            return [fi.reason or "No usable face"]
        problems = []
        if fi.n_faces > 1:
            problems.append("More than one face is visible. Only one person may be in the picture")
        if fi.width < t.min_face_px:
            problems.append("Face is too small. Move closer to the camera")
        if fi.sharpness < t.min_sharpness:
            problems.append("Image is blurry. Hold still and clean the lens")
        if fi.brightness < 45 or fi.brightness > 215:
            problems.append("Lighting is too dark or too bright")
        if fi.pose > t.max_pose:
            problems.append("Face is turned away. Look straight at the camera")
        return problems

    def check_voice(self, vi, enrolling: bool = False) -> list[str]:
        t = self.th
        if not vi.ok:
            return [vi.reason or "No usable speech"]
        problems = []
        need = t.enroll_speech_s if enrolling else t.min_speech_s
        if vi.speech_s < need:
            problems.append(f"Not enough speech ({vi.speech_s:.1f}s heard, {need:.0f}s needed). Speak a full sentence")
        if vi.snr_db < t.min_snr_db:
            problems.append("Too much background noise. Move somewhere quieter")
        if vi.clip_ratio > t.max_clip:
            problems.append("Recording is distorted (too loud). Speak a little further from the microphone")
        return problems
