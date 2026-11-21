"""Nyay Saathi: a free AI legal helper for everyday problems in India (Streamlit app).

Run locally:   streamlit run app.py      (needs GEMINI_API_KEY in env or Streamlit secrets)
Deploy free:   Streamlit Community Cloud, add GEMINI_API_KEY under app Secrets.
"""
import json
import os

import streamlit as st

try:  # Streamlit Cloud secrets -> env, so the backend code stays the same everywhere
    for k in ("GEMINI_API_KEY", "LAWYER_MODEL", "LAWYER_BACKEND", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = st.secrets[k]
except Exception:
    pass

from lawyer.core import DOC_TYPES, KB_BY_ID, Session
from lawyer.llm import LLMError, get_backend

st.set_page_config(page_title="Nyay Saathi: free legal helper", page_icon="⚖️")
st.title("⚖️ Nyay Saathi")
st.caption("Free help for: refund not received · rent deposit not returned · salary not paid. "
           "Type or speak in Hindi, English, Hinglish, Tamil or your language. I am an AI, not a lawyer.")

if "session" not in st.session_state:
    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("LAWYER_BACKEND", "gemini") == "gemini":
        key = st.text_input("Gemini API key (free from aistudio.google.com/apikey)", type="password")
        if not key:
            st.stop()
        os.environ["GEMINI_API_KEY"] = key
    try:
        st.session_state.session = Session(get_backend())
    except LLMError as e:
        st.error(str(e))
        st.stop()
    st.session_state.msgs = []
S = st.session_state.session

with st.sidebar:
    mode = st.radio("Mode", ["advise", "practice"], format_func=lambda m: {
        "advise": "Get help with my problem", "practice": "Practise arguing (AI plays the other side)"}[m])
    st.divider()
    st.subheader("Your case file")
    if S.case:
        st.json(S.case, expanded=False)
        doc = st.selectbox("Prepare a document", list(DOC_TYPES), format_func=lambda d: DOC_TYPES[d])
        if st.button("Draft it"):
            with st.spinner("Drafting..."):
                try:
                    d = S.draft(doc)
                    st.session_state.msgs.append(("assistant", f"**{d['title']}**\n\n```\n{d['body']}\n```"))
                    st.download_button("Download .txt", d["body"], file_name=f"{doc}.txt")
                except LLMError as e:
                    st.error(f"Model error: {e}")
        st.download_button("Download case file (JSON)", json.dumps(S.log, ensure_ascii=False, indent=2),
                           file_name="case_file.json")
    else:
        st.write("Tell me what happened and I'll build it here.")
    st.divider()
    st.caption("Free legal aid: NALSA 15100 · Consumer helpline 1915 · Emergency 112")

for who, txt in st.session_state.msgs:
    st.chat_message(who).markdown(txt)

audio = st.audio_input("🎤 Or record a voice note")
text = st.chat_input("Describe your problem in your own words...")
if audio is not None and st.session_state.get("last_audio") != audio.getvalue()[:64]:
    st.session_state.last_audio = audio.getvalue()[:64]
    with st.spinner("Listening..."):
        try:
            text = S.transcribe(audio.getvalue(), audio.type or "audio/wav")
        except LLMError as e:
            st.error(f"Could not transcribe: {e}")

if text:
    st.session_state.msgs.append(("user", text))
    st.chat_message("user").markdown(text)
    with st.spinner("Thinking..."):
        try:
            out = S.turn(text, mode)
        except LLMError as e:
            st.error(f"Model error, please retry: {e}")
            st.stop()
    body = out["reply"]
    for o in out.get("options", []):
        body += f"\n\n**{o.get('title', '')}**\n" + "\n".join(f"- {s}" for s in o.get("steps", []))
    if out["questions"]:
        body += "\n\n**To help better, tell me:**\n" + "\n".join(f"- {q}" for q in out["questions"])
    if out["cites"]:
        body += "\n\n<sub>Verified sources: " + " · ".join(
            f"[{c}]({KB_BY_ID[c]['source']})" if KB_BY_ID[c]["source"].startswith("http") else c for c in out["cites"]) + "</sub>"
    if out["flags"]:
        body += f"\n\n<sub>Safety checks applied: {', '.join(out['flags'])}</sub>"
    st.session_state.msgs.append(("assistant", body))
    st.chat_message("assistant").markdown(body, unsafe_allow_html=True)
