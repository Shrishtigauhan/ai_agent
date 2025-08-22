# app.py
# Smart Notes – Streamlit + Gemini (Google) + In-memory storage
# --------------------------------------------------------------

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import google.generativeai as genai


# ---------------------------
# Utility data structures
# ---------------------------

@dataclass
class Note:
    id: int
    text: str
    created_iso: str
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None


# ---------------------------
# Gemini helper
# ---------------------------

class GeminiHelper:
    def __init__(
        self,
        api_key: str,
        gen_model_name: str = "gemini-1.5-flash",   # default model
        embed_model_name: str = "text-embedding-004",
        temperature: float = 0.2,                  # default
    ):
        self.api_key = api_key
        self.gen_model_name = gen_model_name
        self.embed_model_name = embed_model_name
        self.temperature = float(temperature)

        genai.configure(api_key=self.api_key)
        self._gen_model = genai.GenerativeModel(model_name=self.gen_model_name)

    def embed(self, text: str) -> List[float]:
        if not text.strip():
            return []
        rsp = genai.embed_content(model=self.embed_model_name, content=text)
        return rsp["embedding"]

    def summarize_and_tag(self, text: str) -> Tuple[str, List[str]]:
        prompt = f"""
You are an expert note-taking assistant. Given the note below, do two things:

1) Produce a concise summary as 5-8 bullet points.
2) Suggest 3-7 topical tags prefixed with "TAGS:".

NOTE:
\"\"\"{text.strip()}\"\"\"
"""
        response = self._gen_model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        raw = (response.text or "").strip()

        tags = []
        tag_match = re.search(r"^TAGS:\s*(.+)$", raw, flags=re.MULTILINE | re.IGNORECASE)
        if tag_match:
            tags_line = tag_match.group(1)
            tags = [t.strip().lower().replace(" ", "-") for t in re.split(r"[;,]", tags_line) if t.strip()]
            raw = re.sub(r"^TAGS:.*$", "", raw, flags=re.MULTILINE).strip()

        return raw, tags

    def answer_question(self, query: str, retrieved_notes: List[Note]) -> str:
        context_blocks = []
        for n in retrieved_notes:
            body = n.summary if n.summary else _truncate(n.text, 1200)
            block = f"(note_id={n.id}, date={n.created_iso}, tags=[{', '.join(n.tags or [])}])\n{body}"
            context_blocks.append(block)
        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "NO_MATCHES"

        prompt = f"""
You are a helpful assistant answering questions based ONLY on the user's saved notes.

CONTEXT:
{context}

USER QUESTION:
{query}
"""
        response = self._gen_model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        return (response.text or "I couldn't find that in your notes.").strip()


# ---------------------------
# Simple in-memory "DB"
# ---------------------------

def _ensure_state():
    if "notes" not in st.session_state:
        st.session_state.notes: List[Note] = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1


def _add_note(text: str, helper: Optional[GeminiHelper], auto_summarize: bool) -> Note:
    created_iso = datetime.now().isoformat(timespec="seconds")
    new = Note(
        id=st.session_state.next_id,
        text=text.strip(),
        created_iso=created_iso,
    )
    st.session_state.next_id += 1

    if helper:
        try:
            new.embedding = helper.embed(new.text)
        except Exception as e:
            st.warning(f"Embedding failed: {e}")

    if helper and auto_summarize:
        try:
            summary, tags = helper.summarize_and_tag(new.text)
            new.summary = summary
            new.tags = tags
        except Exception as e:
            st.warning(f"Summarize/tag failed: {e}")

    st.session_state.notes.insert(0, new)
    return new


def _truncate(s: str, max_chars: int) -> str:
    s = s.strip()
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _retrieve_notes(query: str, helper: Optional[GeminiHelper], top_k: int = 5) -> List[Note]:
    if not helper:
        return []
    try:
        q_emb = np.array(helper.embed(query), dtype=np.float32)
    except Exception as e:
        st.warning(f"Query embedding failed: {e}")
        return []

    scored = []
    for n in st.session_state.notes:
        if not n.embedding:
            try:
                n.embedding = helper.embed(n.text)
            except Exception as e:
                st.warning(f"Embedding note {n.id} failed: {e}")
                n.embedding = []
        n_vec = np.array(n.embedding or [], dtype=np.float32)
        sim = _cosine_sim(q_emb, n_vec)
        scored.append((sim, n))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [n for (sim, n) in scored[: max(1, top_k)] if sim > 0.0]


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Smart Notes (Gemini)", page_icon="📝", layout="wide")
_ensure_state()

# --- No sidebar config ---
api_key = os.getenv("API_KEY")  # must be set in env
helper = GeminiHelper(api_key=api_key) if api_key else None
ai_enabled = helper is not None
auto_summarize_on_save = True
top_k = 5

st.title("📝 Smart Notes")

# 1) Add new note
st.subheader("Add a New Note")
note_text = st.text_area("Type or paste your note here", key="note_draft", height=180)

def clear_draft():
    st.session_state.note_draft = ""

col_save, col_clear = st.columns([1, 1])
with col_save:
    if st.button("Save note", type="primary", use_container_width=True, disabled=(len(note_text.strip()) == 0)):
        created = _add_note(note_text, helper, auto_summarize=auto_summarize_on_save)
        st.success(f"Saved note #{created.id} ({created.created_iso}).")
        st.experimental_rerun()

with col_clear:
    if st.button("Clear draft", use_container_width=True, on_click=clear_draft):
        st.experimental_rerun()

st.divider()

# 2) Ask questions
st.subheader("AI search")
q = st.text_input("Ask in natural language", placeholder="e.g., What were my meeting takeaways last week?")
col_ask, col_show = st.columns([1, 1])

if col_ask.button("Answer", type="primary", use_container_width=True, disabled=not ai_enabled):
    if not st.session_state.notes:
        st.warning("You have no notes yet.")
    elif not ai_enabled:
        st.warning("AI is disabled until API_KEY is set in environment.")
    else:
        with st.spinner("Retrieving notes and thinking..."):
            retrieved = _retrieve_notes(q, helper, top_k=top_k)
            answer = helper.answer_question(q, retrieved)
        st.markdown("#### Answer")
        st.markdown(answer)
        if retrieved:
            with st.expander("Show retrieved notes (context)"):
                for n in retrieved:
                    st.markdown(f"**Note #{n.id} — {n.created_iso}**")
                    if n.tags:
                        st.markdown(f"Tags: `{', '.join(n.tags)}`")
                    st.markdown(f"{n.summary if n.summary else _truncate(n.text, 600)}")
                    st.caption("—")

if col_show.button("Show top matches", use_container_width=True, disabled=not ai_enabled):
    if not st.session_state.notes:
        st.warning("You have no notes yet.")
    elif not ai_enabled:
        st.warning("AI is disabled until API_KEY is set in environment.")
    else:
        retrieved = _retrieve_notes(q or "", helper, top_k=top_k)
        st.markdown("#### Top matches")
        for sim_i, n in enumerate(retrieved, start=1):
            st.markdown(f"- **#{n.id}** ({n.created_iso}) — {(_truncate(n.summary, 140) if n.summary else _truncate(n.text, 140))}")

st.divider()

# 3) All notes display
st.subheader("Your notes")
if not st.session_state.notes:
    st.info("No notes yet — add one above!")
else:
    filt = st.text_input("Quick filter", placeholder="budget, Q4, roadmap, ...")
    for n in st.session_state.notes:
        blob = " ".join([n.text or "", n.summary or "", " ".join(n.tags or []), n.created_iso]).lower()
        if filt and filt.lower() not in blob:
            continue
        with st.expander(f"🗒️ Note #{n.id} — {n.created_iso}", expanded=False):
            st.markdown("**Original text**")
            st.write(n.text)
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.caption("Summary")
                st.write(n.summary or "_No summary yet._")
            with col_b:
                st.caption("Utilities")
                if st.button(f"Summarize & tag #{n.id}", key=f"sum_{n.id}", disabled=not ai_enabled):
                    if not ai_enabled:
                        st.warning("API_KEY must be set.")
                    else:
                        with st.spinner("Summarizing & tagging..."):
                            try:
                                summary, tags = helper.summarize_and_tag(n.text)
                                n.summary, n.tags = summary, tags
                                if not n.embedding:
                                    n.embedding = helper.embed(n.text)
                                st.success("Updated summary & tags.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        st.experimental_rerun()

st.caption("---")
st.caption("Notes With AI")
