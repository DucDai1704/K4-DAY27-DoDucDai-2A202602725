"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Database helper for recent sessions ──────────────────────────────────
async def get_recent_sessions() -> list[dict]:
    """Fetch recent review sessions from audit_events for the sidebar."""
    try:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 15
                """
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


async def load_session_state(thread_id: str):
    """Load latest state and check for interrupts from checkpointer."""
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
            await cp.setup()
            app = build_graph(cp)
            cfg = {"configurable": {"thread_id": thread_id}}
            state = await app.aget_state(cfg)
            int_payload = None
            if state and state.tasks:
                for t in state.tasks:
                    if t.interrupts:
                        int_payload = t.interrupts[0].value
                        break
            values = state.values if state and state.values else None
            return values, int_payload
    except Exception:
        return None, None


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop Pull Request Review system powered by LangGraph, OpenRouter, and SQLite.")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    sessions = asyncio.run(get_recent_sessions())
    if sessions:
        for s in sessions:
            tid = s["thread_id"]
            short_id = tid[:8]
            risk = s.get("worst_risk", "med")
            icon = "🔴" if risk == "high" else ("🟡" if risk == "med" else "🟢")
            url_part = s["pr_url"].split("github.com/")[-1] if "github.com/" in s["pr_url"] else s["pr_url"]

            with st.container(border=True):
                st.markdown(f"**{icon} {url_part}**")
                st.caption(f"Thread: `{short_id}` • Events: {s['events']}")
                st.caption(f"Last: {str(s['last_event'])[:19]}")
                if st.button("Load Session", key=f"btn_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    values, int_payload = asyncio.run(load_session_state(tid))
                    if int_payload is not None:
                        st.session_state.interrupt_payload = int_payload
                    elif values is not None:
                        st.session_state.final = values
                    st.rerun()
    else:
        st.info("No recorded sessions yet. Run a review to populate the audit log!")

    st.divider()
    st.markdown("### 💡 Quick Test PRs")
    if st.button("PR #1: Task priority (58–72%)", use_container_width=True):
        st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
        st.rerun()
    if st.button("PR #2: Login & sync (<58%)", use_container_width=True):
        st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
        st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        pr_url = st.text_input(
            "PR URL",
            value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
    with col_btn:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Run review", type="primary", use_container_width=True)


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    with st.container(border=True):
        st.markdown(f"### ⚠️ Human Approval Requested — Confidence: **{conf:.0%}**")
        st.progress(conf)
        st.info(f"**Reasoning:** {payload['confidence_reasoning']}")

        st.markdown("#### 📝 Summary")
        st.markdown(payload["summary"])

        comments = payload.get("comments", [])
        if comments:
            st.markdown(f"#### 🔍 Review Comments ({len(comments)})")
            for c in comments:
                sev = c.get("severity", "issue")
                sev_icon = "🛑" if sev == "blocker" else ("⚠️" if sev == "issue" else "💡")
                st.markdown(f"{sev_icon} **[{sev.upper()}]** `{c.get('file')}:{c.get('line') or '?'}` — {c.get('body')}")

        with st.expander("📄 View Unified Diff Preview"):
            st.code(payload.get("diff_preview", ""), language="diff")

        st.markdown("---")
        feedback = st.text_input("Reviewer Notes / Feedback (optional):", key="approval_feedback")

        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Approve & Post Comment", type="primary", use_container_width=True):
            return {"choice": "approve", "feedback": feedback}
        if col2.button("❌ Reject Review", use_container_width=True):
            return {"choice": "reject", "feedback": feedback}
        if col3.button("✏️ Edit / Request Refinement", use_container_width=True):
            return {"choice": "edit", "feedback": feedback}

    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    with st.container(border=True):
        st.error(f"### 🚨 Strong Escalation Triggered — Confidence: **{conf:.0%}**")
        st.progress(conf)
        st.caption(f"**Reasoning:** {payload['confidence_reasoning']}")

        if payload.get("risk_factors"):
            st.warning("**Identified Risk Factors:**\n" + "\n".join(f"- {rf}" for rf in payload["risk_factors"]))

        st.markdown("#### 📝 Summary")
        st.markdown(payload["summary"])

        st.markdown("---")
        st.markdown("#### ❓ Reviewer Clarification Required")
        st.caption("The agent has detected significant uncertainties. Please answer the following questions to help synthesize an accurate review:")

        with st.form("escalation_form"):
            answers: dict[str, str] = {}
            questions = payload.get("questions", [])
            for idx, q in enumerate(questions):
                st.markdown(f"**Q{idx + 1}: {q}**")
                answers[q] = st.text_area(f"Your answer to Q{idx + 1}", key=f"q_input_{idx}", height=70)

            submitted_ans = st.form_submit_button("📤 Submit Answers & Synthesize Review", type="primary", use_container_width=True)
            if submitted_ans:
                return answers

    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)

        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and running LLM analysis..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent with your input..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    with st.container(border=True):
        if action.startswith("auto") or action.startswith("committed"):
            st.balloons()
            st.success(f"🎉 **Status:** `{action}` — Review comment successfully posted to GitHub!")
        elif action == "rejected":
            st.warning(f"⚠️ **Status:** `rejected` — Review rejected by human reviewer. No comments posted.")
        else:
            st.info(f"ℹ️ **Status:** `final_action = {action}`")

        if "analysis" in final:
            a = final["analysis"]
            st.markdown(f"**Final Confidence:** `{a.confidence:.0%}`")
            st.markdown(f"**Summary:** {a.summary}")

        st.markdown(f"🔗 [Open Pull Request on GitHub]({st.session_state.pr_url})")
        st.caption(
            f"Session Thread ID: `{st.session_state.thread_id}`  •  "
            f"Replay command: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
        )
