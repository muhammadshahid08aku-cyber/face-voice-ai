# Smart Multi-Agent Face + Voice Verification

Consent-based identity verification. An **Orchestrator** coordinates specialist agents:
Consent & Privacy, Guardrail, Face (YuNet + SFace), Voice (WeSpeaker via sherpa-onnx), Quality, Fusion & Decision, Audit, Explainer.
Decisions are rule-based (ACCEPT / REVIEW / REJECT / RETRY / LOCKED) and every one is explained and logged.

## Run locally
    pip install -r requirements.txt
    ADMIN_PASSWORD=choose-one streamlit run app.py
Models (~40 MB) download on first start and are SHA-256 verified. Local data goes to `biometrics.db`.

## Deploy (Streamlit Community Cloud + Supabase)
1. Put this folder in a GitHub repo (the `.gitignore` already excludes keys, databases and models).
2. Supabase: create a project, copy the **Session pooler** connection string.
3. Generate an encryption key:  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
   Keep it safe: lose it and all templates become unreadable.
4. share.streamlit.io -> New app -> choose repo, main file `app.py` -> Advanced settings -> Secrets: paste
   `DATABASE_URL`, `BIOMETRIC_KEY`, `ADMIN_PASSWORD` (see `.streamlit/secrets.toml.example`).

## Privacy
Explicit consent + typed name; only encrypted numeric templates stored (never photos/audio); 365-day retention;
erase on request; audit log holds no biometrics; lockout after 5 failures in 10 minutes.

## Limitations (important)
- **No liveness / anti-spoofing**: a printed photo or voice replay can fool it. Do not protect money, health data or doors with it alone.
- Face scores separated cleanly in testing (genuine >= 0.94, impostors <= 0.26). Voice tests used synthetic voices only, so voice error rates are unmeasured on real people. Calibrate thresholds on your own users (env vars such as `VOICE_ACCEPT=0.7`).
- Default policy is `strict` (both modalities must pass).
- Model licences (trained on VoxCeleb / MS1M data) may restrict commercial use; check before selling.
- Follow local biometric and privacy law; use only with informed consent.
