# app.py
from __future__ import annotations

import os
import re
import time
import uuid
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "Vocab Buddy"
DB_PATH = os.environ.get("VOCAB_APP_DB", "vocab_app.sqlite3")


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class VocabItem:
    word: str
    meaning: str
    example: str = ""
    tags: str = ""


DEFAULT_VOCAB = [
    VocabItem("ubiquitous", "present everywhere", "Smartphones are ubiquitous nowadays.", "adjective"),
    VocabItem("mitigate", "make less severe, serious, or painful", "Trees help mitigate air pollution.", "verb"),
    VocabItem("meticulous", "showing great attention to detail", "She kept meticulous notes.", "adjective"),
    VocabItem("concur", "agree", "I concur with your assessment.", "verb"),
    VocabItem("pragmatic", "dealing with things sensibly and realistically", "They took a pragmatic approach.", "adjective"),
]


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s'-]", "", s)  # keep letters/digits/underscore + space + ' and -
    return s


def ensure_session_ids() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())  # pseudonymous by default
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def pick_distractors(items: List[VocabItem], correct_idx: int, k: int = 3) -> List[int]:
    idxs = list(range(len(items)))
    idxs.remove(correct_idx)
    random.shuffle(idxs)
    return idxs[: min(k, len(idxs))]


# -----------------------------
# Database (SQLite) logging
# -----------------------------
@st.cache_resource
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            mode TEXT NOT NULL,
            word TEXT,
            prompt TEXT,
            response TEXT,
            correct INTEGER,
            latency_ms INTEGER,
            extra_json TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            user_id TEXT NOT NULL,
            word TEXT NOT NULL,
            box INTEGER NOT NULL DEFAULT 1,
            seen_count INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            last_seen_utc TEXT,
            last_result INTEGER,
            PRIMARY KEY (user_id, word)
        );
        """
    )
    conn.commit()
    return conn


def log_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    mode: str,
    word: Optional[str] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    correct: Optional[bool] = None,
    latency_ms: Optional[int] = None,
    extra_json: str = "",
) -> None:
    ensure_session_ids()
    conn.execute(
        """
        INSERT INTO events (id, ts_utc, user_id, session_id, event_type, mode, word, prompt, response, correct, latency_ms, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            utc_now_iso(),
            st.session_state.user_id,
            st.session_state.session_id,
            event_type,
            mode,
            word,
            prompt,
            response,
            None if correct is None else int(bool(correct)),
            latency_ms,
            extra_json,
        ),
    )
    conn.commit()


def upsert_progress(conn: sqlite3.Connection, user_id: str, word: str, correct: bool) -> None:
    # Simple Leitner-ish update:
    # - If correct: move up one box (max 5)
    # - If wrong: drop to box 1
    row = conn.execute(
        "SELECT box, seen_count, correct_count FROM progress WHERE user_id=? AND word=?",
        (user_id, word),
    ).fetchone()

    if row is None:
        box = 2 if correct else 1
        seen = 1
        corr = 1 if correct else 0
        conn.execute(
            """
            INSERT INTO progress (user_id, word, box, seen_count, correct_count, last_seen_utc, last_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, word, box, seen, corr, utc_now_iso(), int(bool(correct))),
        )
    else:
        old_box, seen, corr = row
        new_box = min(5, old_box + 1) if correct else 1
        conn.execute(
            """
            UPDATE progress
            SET box=?, seen_count=?, correct_count=?, last_seen_utc=?, last_result=?
            WHERE user_id=? AND word=?
            """,
            (new_box, seen + 1, corr + (1 if correct else 0), utc_now_iso(), int(bool(correct)), user_id, word),
        )
    conn.commit()


def get_user_progress(conn: sqlite3.Connection, user_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT word, box, seen_count, correct_count, last_seen_utc, last_result
        FROM progress
        WHERE user_id=?
        ORDER BY box ASC, seen_count DESC, word ASC
        """,
        conn,
        params=(user_id,),
    )
    return df


def read_events(conn: sqlite3.Connection, user_id: Optional[str] = None) -> pd.DataFrame:
    if user_id:
        return pd.read_sql_query(
            "SELECT * FROM events WHERE user_id=? ORDER BY ts_utc DESC",
            conn,
            params=(user_id,),
        )
    return pd.read_sql_query("SELECT * FROM events ORDER BY ts_utc DESC", conn)


# -----------------------------
# Vocab loading
# -----------------------------
def vocab_from_dataframe(df: pd.DataFrame) -> List[VocabItem]:
    # expected columns: word, meaning; optional: example, tags
    cols = {c.lower(): c for c in df.columns}
    if "word" not in cols or "meaning" not in cols:
        raise ValueError("CSV must have at least columns: word, meaning")

    items: List[VocabItem] = []
    for _, row in df.iterrows():
        w = str(row[cols["word"]]).strip()
        m = str(row[cols["meaning"]]).strip()
        if not w or not m or w.lower() == "nan" or m.lower() == "nan":
            continue
        ex = str(row[cols["example"]]).strip() if "example" in cols else ""
        tg = str(row[cols["tags"]]).strip() if "tags" in cols else ""
        items.append(VocabItem(w, m, ex if ex.lower() != "nan" else "", tg if tg.lower() != "nan" else ""))
    return items or DEFAULT_VOCAB


@st.cache_data
def load_vocab_cached(csv_bytes: Optional[bytes]) -> List[VocabItem]:
    if csv_bytes:
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        return vocab_from_dataframe(df)
    return DEFAULT_VOCAB


# -----------------------------
# UI components
# -----------------------------
def render_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="centered")
    st.title("📚 Vocab Buddy")
    st.caption("Flashcards + quizzes with built-in (pseudonymous) learning analytics.")


def sidebar_controls() -> Tuple[str, str, Optional[bytes]]:
    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Mode", ["Learn (Flashcards)", "Quiz (MCQ)", "Quiz (Type)", "Add/Upload Vocab", "Analytics"], index=0)

        st.divider()
        st.subheader("Learner ID")
        st.write("Set a nickname (optional). Your raw UUID stays in the logs unless you overwrite it.")
        nickname = st.text_input("Nickname", value=st.session_state.get("nickname", ""), placeholder="e.g., Yiran")
        if nickname != st.session_state.get("nickname", ""):
            st.session_state.nickname = nickname

        if st.button("New session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.success("New session started.")

        st.divider()
        st.subheader("Vocab source")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Columns: word, meaning (optional: example, tags)")
        csv_bytes = uploaded.getvalue() if uploaded else None

        privacy = st.toggle("Hide raw responses in logs", value=st.session_state.get("hide_responses", False))
        st.session_state.hide_responses = privacy

        return mode, nickname, csv_bytes


def init_learning_state(items: List[VocabItem]) -> None:
    if "learn_idx" not in st.session_state:
        st.session_state.learn_idx = 0
    if "order" not in st.session_state or len(st.session_state.order) != len(items):
        order = list(range(len(items)))
        random.shuffle(order)
        st.session_state.order = order
        st.session_state.learn_idx = 0

    if "quiz_current" not in st.session_state:
        st.session_state.quiz_current = None
    if "quiz_started_at" not in st.session_state:
        st.session_state.quiz_started_at = None


def next_flashcard(items: List[VocabItem]) -> VocabItem:
    idx = st.session_state.order[st.session_state.learn_idx % len(items)]
    st.session_state.learn_idx += 1
    return items[idx]


def choose_next_quiz_word(conn: sqlite3.Connection, items: List[VocabItem]) -> int:
    """Bias towards lower boxes / unseen words."""
    ensure_session_ids()
    user_id = st.session_state.user_id

    prog = get_user_progress(conn, user_id)
    box_map = {r["word"]: int(r["box"]) for _, r in prog.iterrows()} if not prog.empty else {}

    scores = []
    for i, it in enumerate(items):
        box = box_map.get(it.word, 1)
        # lower box => higher weight
        weight = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}.get(box, 3)
        scores.append((i, weight))

    population = [i for i, _ in scores]
    weights = [w for _, w in scores]
    return random.choices(population, weights=weights, k=1)[0]


# -----------------------------
# Pages
# -----------------------------
def page_learn(conn: sqlite3.Connection, items: List[VocabItem]):
    st.subheader("Learn (Flashcards)")
    st.write("Flip through words. Use this mode for exposure; quizzes update progress boxes.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Next card ➜", use_container_width=True):
            card = next_flashcard(items)
            st.session_state.current_card = card
            log_event(conn, event_type="view_card", mode="learn", word=card.word, prompt=card.word)

    with col2:
        if st.button("Shuffle deck 🔀", use_container_width=True):
            order = list(range(len(items)))
            random.shuffle(order)
            st.session_state.order = order
            st.session_state.learn_idx = 0
            log_event(conn, event_type="shuffle", mode="learn")

    card: VocabItem = st.session_state.get("current_card", items[st.session_state.order[0]])

    with st.container(border=True):
        st.markdown(f"### {card.word}")
        st.markdown(f"**Meaning:** {card.meaning}")
        if card.example:
            st.markdown(f"**Example:** _{card.example}_")
        if card.tags:
            st.caption(f"tags: {card.tags}")

    progress = (st.session_state.learn_idx % max(1, len(items))) / max(1, len(items))
    st.progress(progress, text=f"Deck position: {st.session_state.learn_idx} (shuffled order)")


def page_quiz_mcq(conn: sqlite3.Connection, items: List[VocabItem]):
    st.subheader("Quiz (Multiple choice)")
    st.write("Choose the correct meaning. Your progress updates like a simple Leitner system (boxes 1–5).")

    if st.session_state.quiz_current is None:
        idx = choose_next_quiz_word(conn, items)
        st.session_state.quiz_current = idx
        st.session_state.quiz_started_at = time.perf_counter()
        log_event(conn, event_type="quiz_start", mode="quiz_mcq", word=items[idx].word, prompt=items[idx].word)

    idx = int(st.session_state.quiz_current)
    item = items[idx]

    distractor_idxs = pick_distractors(items, idx, k=3)
    options = [(idx, item.meaning)] + [(j, items[j].meaning) for j in distractor_idxs]
    random.shuffle(options)

    st.markdown(f"### {item.word}")
    choice = st.radio("Pick the meaning:", options=[m for _, m in options], index=None)

    cols = st.columns([1, 1, 1])
    with cols[0]:
        submit = st.button("Submit ✅", use_container_width=True)
    with cols[1]:
        skip = st.button("Skip ⏭️", use_container_width=True)
    with cols[2]:
        reveal = st.button("Reveal 👀", use_container_width=True)

    if reveal:
        st.info(f"**Correct meaning:** {item.meaning}")
        if item.example:
            st.caption(f"Example: {item.example}")

    if skip:
        log_event(conn, event_type="quiz_skip", mode="quiz_mcq", word=item.word, prompt=item.word)
        st.session_state.quiz_current = None
        st.session_state.quiz_started_at = None
        st.rerun()

    if submit:
        if choice is None:
            st.warning("Pick an option first.")
            return

        latency_ms = int((time.perf_counter() - (st.session_state.quiz_started_at or time.perf_counter())) * 1000)
        correct = (normalise(choice) == normalise(item.meaning))

        response_to_store = "" if st.session_state.get("hide_responses", False) else choice
        log_event(
            conn,
            event_type="quiz_submit",
            mode="quiz_mcq",
            word=item.word,
            prompt=item.word,
            response=response_to_store,
            correct=correct,
            latency_ms=latency_ms,
        )
        upsert_progress(conn, st.session_state.user_id, item.word, correct)

        if correct:
            st.success(f"Correct! ({latency_ms} ms)")
        else:
            st.error(f"Not quite. Correct answer: {item.meaning} ({latency_ms} ms)")

        st.session_state.quiz_current = None
        st.session_state.quiz_started_at = None
        st.rerun()


def page_quiz_type(conn: sqlite3.Connection, items: List[VocabItem]):
    st.subheader("Quiz (Type)")
    st.write("Type the meaning (free text). This is strict-ish but ignores punctuation and extra spaces.")

    if st.session_state.quiz_current is None:
        idx = choose_next_quiz_word(conn, items)
        st.session_state.quiz_current = idx
        st.session_state.quiz_started_at = time.perf_counter()
        log_event(conn, event_type="quiz_start", mode="quiz_type", word=items[idx].word, prompt=items[idx].word)

    idx = int(st.session_state.quiz_current)
    item = items[idx]

    st.markdown(f"### {item.word}")
    typed = st.text_input("Type the meaning:", key=f"typed_{idx}_{st.session_state.session_id}")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        submit = st.button("Submit ✅", use_container_width=True)
    with cols[1]:
        skip = st.button("Skip ⏭️", use_container_width=True)
    with cols[2]:
        hint = st.button("Hint 💡", use_container_width=True)

    if hint:
        # lightweight hint: first 40% of the meaning
        n = max(1, int(len(item.meaning) * 0.4))
        st.info(f"Hint: **{item.meaning[:n]}…**")

    if skip:
        log_event(conn, event_type="quiz_skip", mode="quiz_type", word=item.word, prompt=item.word)
        st.session_state.quiz_current = None
        st.session_state.quiz_started_at = None
        st.rerun()

    if submit:
        latency_ms = int((time.perf_counter() - (st.session_state.quiz_started_at or time.perf_counter())) * 1000)

        # Normalised exact match; you can loosen this if desired.
        correct = normalise(typed) == normalise(item.meaning)

        response_to_store = "" if st.session_state.get("hide_responses", False) else typed
        log_event(
            conn,
            event_type="quiz_submit",
            mode="quiz_type",
            word=item.word,
            prompt=item.word,
            response=response_to_store,
            correct=correct,
            latency_ms=latency_ms,
        )
        upsert_progress(conn, st.session_state.user_id, item.word, correct)

        if correct:
            st.success(f"Correct! ({latency_ms} ms)")
        else:
            st.error(f"Not quite. Model answer: {item.meaning} ({latency_ms} ms)")
            if item.example:
                st.caption(f"Example: {item.example}")

        st.session_state.quiz_current = None
        st.session_state.quiz_started_at = None
        st.rerun()


def page_add_upload(conn: sqlite3.Connection, items: List[VocabItem]):
    st.subheader("Add/Upload Vocab")
    st.write(
        "Upload a CSV in the sidebar or add a few items here (in-memory). "
        "If you want persistence, save your list as CSV and upload it next time."
    )

    st.markdown("**Current items (preview):**")
    st.dataframe(pd.DataFrame([it.__dict__ for it in items]).head(25), use_container_width=True)

    st.divider()
    st.markdown("### Add a new word (in this session)")
    with st.form("add_word_form", clear_on_submit=True):
        w = st.text_input("Word")
        m = st.text_input("Meaning")
        ex = st.text_input("Example (optional)")
        tg = st.text_input("Tags (optional)")
        submitted = st.form_submit_button("Add")

    if submitted:
        if not w.strip() or not m.strip():
            st.warning("Word and meaning are required.")
            return
        new_item = VocabItem(w.strip(), m.strip(), ex.strip(), tg.strip())

        # store in session (won't affect cached list automatically)
        added = st.session_state.get("added_items", [])
        added.append(new_item)
        st.session_state.added_items = added
        log_event(conn, event_type="add_word", mode="add_vocab", word=new_item.word, prompt=new_item.word)
        st.success("Added. (Tip: export your combined list below.)")

    combined = items + st.session_state.get("added_items", [])
    df = pd.DataFrame([it.__dict__ for it in combined])

    st.divider()
    st.markdown("### Export combined vocab as CSV")
    st.download_button(
        "Download vocab.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vocab.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_analytics(conn: sqlite3.Connection, items: List[VocabItem]):
    st.subheader("Analytics")
    ensure_session_ids()

    user_only = st.toggle("Show only my data", value=True)
    user_id = st.session_state.user_id if user_only else None

    events = read_events(conn, user_id=user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Events logged", int(len(events)))
    with col2:
        quizzes = events[events["event_type"] == "quiz_submit"] if not events.empty else pd.DataFrame()
        acc = (quizzes["correct"].mean() * 100) if not quizzes.empty else 0
        st.metric("Quiz accuracy", f"{acc:.1f}%")
    with col3:
        st.metric("Current session", st.session_state.session_id[:8])

    st.divider()
    st.markdown("### Progress (Leitner boxes)")
    prog = get_user_progress(conn, st.session_state.user_id)
    if prog.empty:
        st.info("No quiz progress yet. Do a few quiz items first.")
    else:
        prog2 = prog.copy()
        prog2["accuracy"] = (prog2["correct_count"] / prog2["seen_count"]).round(2)
        st.dataframe(prog2, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Raw event log")
    if events.empty:
        st.info("No events logged yet.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)

        st.download_button(
            "Download event log (CSV)",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name="vocab_events.csv",
            mime="text/csv",
            use_container_width=True,
        )


# -----------------------------
# Main
# -----------------------------
def main():
    render_header()
    ensure_session_ids()

    mode, nickname, csv_bytes = sidebar_controls()

    conn = get_conn(DB_PATH)
    items = load_vocab_cached(csv_bytes)
    added = st.session_state.get("added_items", [])
    all_items = items + added

    init_learning_state(all_items)

    # Identify user in logs (optional nickname)
    if nickname:
        # One-time log per session (rough)
        if not st.session_state.get("logged_nickname", False):
            log_event(conn, event_type="set_nickname", mode="meta", response=nickname if not st.session_state.hide_responses else "")
            st.session_state.logged_nickname = True

    if len(all_items) < 2:
        st.warning("Please add/upload at least 2 vocab items to run quizzes.")
        page_add_upload(conn, all_items)
        return

    if mode == "Learn (Flashcards)":
        page_learn(conn, all_items)
    elif mode == "Quiz (MCQ)":
        page_quiz_mcq(conn, all_items)
    elif mode == "Quiz (Type)":
        page_quiz_type(conn, all_items)
    elif mode == "Add/Upload Vocab":
        page_add_upload(conn, all_items)
    elif mode == "Analytics":
        page_analytics(conn, all_items)

    st.caption(f"Data store: `{DB_PATH}` • user_id: `{st.session_state.user_id[:8]}…`")


if __name__ == "__main__":
    main()
