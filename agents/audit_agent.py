"""Audit Agent: one place that writes the audit log. It never receives raw photos, audio or embeddings."""


class AuditAgent:
    name = "AuditAgent"

    def __init__(self, store):
        self.store = store

    def record(self, actor, action, person_id=None, decision=None, face=None, voice=None, detail="", session=None):
        self.store.log(actor, action, person_id=person_id, decision=decision,
                       face=None if face is None else round(float(face), 4),
                       voice=None if voice is None else round(float(voice), 4),
                       detail=detail, session=session)
