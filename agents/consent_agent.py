"""Consent & Privacy Agent: no enrollment without explicit consent. Only numeric templates are kept."""
import datetime as dt

CONSENT_VERSION = "2026-11"


def consent_text(retention_days: int) -> str:
    return (
        "I agree that my face photos and voice recordings are converted into numeric templates, "
        f"stored encrypted, and used only to check my identity in this system. Templates are kept for up to "
        f"{retention_days} days, and I can ask an administrator to delete them at any time. "
        "The original photos and recordings are not saved."
    )


class ConsentPrivacyAgent:
    name = "ConsentPrivacyAgent"

    def __init__(self, th):
        self.th = th

    def validate(self, name: str, consent: bool, typed_name: str) -> list[str]:
        problems = []
        if not (name or "").strip():
            problems.append("Enter the person's name")
        if not consent:
            problems.append("The person has not ticked the consent box")
        if (name or "").strip().lower() != (typed_name or "").strip().lower():
            problems.append("The typed confirmation name does not match")
        return problems

    def retention_until(self):
        return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(days=self.th.retention_days)
