"""Smart Multi-Agent Face + Voice Verification  (Streamlit UI)"""
import hmac
import uuid

import streamlit as st

import settings
from agents.consent_agent import consent_text
from orchestrator import Orchestrator
from settings import load_thresholds
from store import Store, load_key

st.set_page_config(page_title="Multi-Agent Face & Voice ID", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading models (first start downloads about 40 MB)...")
def boot():
    url = settings.get("DATABASE_URL", "sqlite:///biometrics.db")
    store = Store(url, load_key(url, settings.get("BIOMETRIC_KEY")))
    th = load_thresholds()
    orch = Orchestrator(store, th)
    orch.purge()
    return orch


try:
    orch = boot()
except Exception as e:  # show a readable setup error instead of a stack trace
    st.error(f"Setup problem: {e}")
    st.stop()

store, th = orch.store, orch.th
st.session_state.setdefault("sid", uuid.uuid4().hex[:12])
sid = st.session_state["sid"]
BADGE = {"ACCEPT": "🟢", "REVIEW": "🟠", "REJECT": "🔴", "RETRY": "🟡", "LOCKED": "🔒", "OK": "🟢", "ERROR": "🔴"}


def is_admin() -> bool:
    return st.session_state.get("admin", False)


def admin_gate():
    if is_admin():
        return True
    pw = settings.get("ADMIN_PASSWORD")
    if not pw:
        st.warning("Set ADMIN_PASSWORD in secrets to use admin pages.")
        return False
    with st.form("login"):
        entered = st.text_input("Administrator password", type="password")
        if st.form_submit_button("Sign in") and hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["admin"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password")
    return False


def show_outcome(o):
    box = {"ACCEPT": st.success, "OK": st.success, "REVIEW": st.warning, "RETRY": st.warning}.get(o.label, st.error)
    box(f"{BADGE.get(o.label, '')} **{o.label}** - {o.message}")
    c = st.columns(3)
    if o.face_score is not None:
        c[0].metric("Face similarity", f"{o.face_score:.2f}")
    if o.voice_score is not None:
        c[1].metric("Voice similarity", f"{o.voice_score:.2f}")
    if o.combined is not None:
        c[2].metric("Combined (1.0 = accept)", f"{o.combined:.2f}")
    if o.ranking:
        st.caption("Closest candidates")
        st.dataframe(o.ranking, hide_index=True)
    with st.expander("Agent trace (what each agent did)"):
        st.dataframe([{"agent": t["agent"], "status": t["status"], "detail": t["detail"], "ms": t["ms"]} for t in o.trace],
                     hide_index=True)


def read(f):
    return f.getvalue() if f is not None else None


page = st.sidebar.radio("Menu", ["Verify", "Enroll (admin)", "People & audit (admin)", "About & privacy"])
if is_admin() and st.sidebar.button("Sign out"):
    st.session_state["admin"] = False
    st.rerun()

# ------------------------------------------------------------------ Verify
if page == "Verify":
    st.title("🛡️ Verify identity")
    st.caption("Face + voice are checked by separate AI agents, then a decision agent combines them. "
               "No photo or audio is saved.")
    mode = st.radio("Mode", ["I have an ID code (1:1)", "Find who I am (1:N)"], horizontal=True)
    ext = st.text_input("ID code (like U-1A2B3C)") if mode.startswith("I have") else None
    use_face = st.toggle("Use face", True)
    use_voice = st.toggle("Use voice", True)
    left, right = st.columns(2)
    img = left.camera_input("Look at the camera") if use_face else None
    aud = right.audio_input("Say: 'My voice is my password, please verify me.' (3+ seconds)") if use_voice else None
    if st.button("Check", type="primary"):
        if not (img or aud):
            st.warning("Provide a photo and/or a recording.")
        elif ext is not None and not ext.strip():
            st.warning("Enter your ID code.")
        else:
            with st.spinner("Agents are working..."):
                if ext is not None:
                    o = orch.verify(ext, read(img), read(aud), session=sid)
                else:
                    o = orch.identify(read(img), read(aud), session=sid)
            show_outcome(o)
            if o.person:
                st.balloons()

# ------------------------------------------------------------------ Enroll
elif page == "Enroll (admin)":
    st.title("Enroll a person")
    if admin_gate():
        st.info(consent_text(th.retention_days))
        name = st.text_input("Full name")
        consent = st.checkbox("The person has read the statement above and agrees")
        typed = st.text_input("Person types their full name again to confirm")
        st.write(f"**Face photos** - at least {th.min_face_samples}, slightly different angles/lighting")
        n_f = st.session_state.setdefault("n_f", th.min_face_samples)
        faces = [st.camera_input(f"Photo {i + 1}", key=f"ph{i}") for i in range(n_f)]
        st.write(f"**Voice recordings** - at least {th.min_voice_samples}, each {th.enroll_speech_s:.0f}+ seconds of speech")
        n_v = st.session_state.setdefault("n_v", th.min_voice_samples)
        voices = [st.audio_input(f"Recording {i + 1}", key=f"au{i}") for i in range(n_v)]
        dup = st.checkbox("Allow duplicate (same person enrolled twice)", False)
        if st.button("Enroll", type="primary"):
            with st.spinner("Checking quality, consent and duplicates..."):
                o = orch.enroll(name, consent, typed, [read(f) for f in faces if f], [read(v) for v in voices if v],
                                allow_duplicate=dup)
            show_outcome(o)
            for p in o.problems:
                st.write("•", p)
            if o.label == "OK":
                st.code(o.person["ext_id"])
                st.caption("Give this ID code to the person. Enrollment photos and audio were not stored.")

# ------------------------------------------------------------------ Admin
elif page == "People & audit (admin)":
    st.title("People & audit log")
    if admin_gate():
        ppl = store.persons()
        st.subheader(f"Enrolled people ({len(ppl)})")
        if ppl:
            st.dataframe([{k: p[k] for k in ("ext_id", "name", "consent_version", "consent_at", "retention_until")} for p in ppl],
                         hide_index=True)
            pick = st.selectbox("Erase a person's biometric data", [f"{p['ext_id']} - {p['name']}" for p in ppl])
            if st.checkbox("I understand this deletes their templates permanently") and st.button("Erase now"):
                orch.erase(next(p["id"] for p in ppl if pick.startswith(p["ext_id"])))
                st.success("Erased.")
                st.rerun()
        st.subheader("Active thresholds")
        st.json({k: v for k, v in vars(th).items()})
        st.caption("Change them with environment variables or secrets (upper-case names, e.g. VOICE_ACCEPT=0.7).")
        st.subheader("Audit log (no biometric data inside)")
        st.dataframe(store.audit_df(), hide_index=True)

# ------------------------------------------------------------------ About
else:
    st.title("About & privacy")
    st.markdown(f"""
**Agents:** Orchestrator · Consent & Privacy · Guardrail · Face · Voice · Quality · Fusion & Decision · Audit · Explainer.

**Privacy:** explicit consent before enrollment; only numeric templates are stored, encrypted; originals are never saved;
templates expire after {th.retention_days} days; an administrator can erase anyone at any time; every decision is logged
without biometric data; repeated failures lock the kiosk/account for {th.lock_minutes} minutes.

**Limits - please read:**
- There is **no liveness / anti-spoofing**. A printed photo or a recording of someone's voice can fool it.
  Do not use it alone to protect money, health records or doors.
- Accuracy depends on lighting, microphone and background noise. Borderline cases go to **REVIEW** for a human.
- Face and voice models are research-licensed in part; check licences before commercial use.
- Use only with the informed consent of the people involved, and follow your local biometric/privacy law.
""")
