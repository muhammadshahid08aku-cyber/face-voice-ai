"""Guardrail Agent: input limits and a lock-out after repeated failed attempts."""


class GuardrailAgent:
    name = "GuardrailAgent"

    def __init__(self, th, store):
        self.th = th
        self.store = store

    def check_inputs(self, image: bytes | None, audio: bytes | None) -> list[str]:
        problems = []
        if image is not None and len(image) > self.th.max_image_mb * 1e6:
            problems.append(f"Image is larger than {self.th.max_image_mb:g} MB")
        if audio is not None and len(audio) > self.th.max_audio_mb * 1e6:
            problems.append(f"Audio is larger than {self.th.max_audio_mb:g} MB")
        if image is None and audio is None:
            problems.append("Provide a photo, a voice recording, or both")
        return problems

    def locked(self, session: str | None, person_id: int | None = None) -> bool:
        n_s = self.store.failures(session=session, minutes=self.th.lock_minutes) if session else 0
        n_p = self.store.failures(person_id=person_id, minutes=self.th.lock_minutes) if person_id else 0
        return max(n_s, n_p) >= self.th.lock_failures
