"""Explainer Agent: plain-language reasons for every decision. Rule-based, so it never invents anything."""


class ExplainerAgent:
    name = "ExplainerAgent"

    def explain(self, label, face, voice, decision, problems=None, locked=False, person_name=None):
        if locked:
            return "Too many failed attempts. This kiosk or account is locked for a few minutes. Ask an administrator."
        if label == "RETRY":
            return "The sample was not good enough to compare. " + "; ".join(problems or []) + "."
        bits = []
        if face is not None:
            bits.append(f"face similarity {face:.2f}")
        if voice is not None:
            bits.append(f"voice similarity {voice:.2f}")
        evidence = ", ".join(bits) or "no evidence"
        extra = (" " + " ".join(decision.reasons)) if decision and decision.reasons else ""
        if label == "ACCEPT":
            who = f" Welcome, {person_name}." if person_name else ""
            return f"Accepted: {evidence} both met the confidence levels.{who}{extra}"
        if label == "REVIEW":
            return f"Needs a human check: {evidence} was close to, but below, the confidence levels.{extra}"
        return f"Rejected: {evidence} did not match.{extra}"
