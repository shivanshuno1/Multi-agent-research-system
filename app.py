"""
Streamlit frontend for Multi-agent-research-system
https://github.com/shivanshuno1/Multi-agent-research-system

Drop this file into the root of the repo (next to agents.py, pipeline.py,
tools.py) and run:  streamlit run app.py

Deploy on Streamlit Community Cloud:
  1. Push this file + requirements.txt to the repo.
  2. Go to share.streamlit.io -> New app -> point at this repo/app.py.
  3. In the app's Settings -> Secrets, add:
       GEMINI_API_KEY = "..."
       TAVILY_API_KEY = "..."
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# API keys: prefer st.secrets (used on Streamlit Cloud), fall back to env
# vars, and let the user type them in as a last resort. These must be set
# BEFORE pipeline/agents are imported, since agents.py reads them at
# import time when it builds the ChatGoogleGenerativeAI client.
# ---------------------------------------------------------------------------
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

_secret_gemini = os.getenv("GEMINI_API_KEY")
_secret_tavily = os.getenv("TAVILY_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")

    if _secret_gemini and _secret_tavily:
        # Keys are already configured server-side (e.g. Render/Streamlit Cloud
        # secrets). Never echo the real value back into an input field, since
        # that would leak it into the page's HTML even with type="password".
        st.success("API keys loaded from server configuration.")
        gemini_key, tavily_key = _secret_gemini, _secret_tavily
    else:
        st.caption(
            "No server-side secrets found — enter keys for this session only. "
            "They are not stored or logged, but avoid using this on a "
            "publicly shared deployment."
        )
        gemini_key = st.text_input("Gemini API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")

    st.divider()
    st.markdown(
        "**Pipeline stages**\n"
        "1. 🔍 Search agent (Tavily)\n"
        "2. 📖 Reader agent (scrapes top result)\n"
        "3. ✍️ Writer chain (drafts report)\n"
        "4. 🧐 Critic chain (scores the report)"
    )
    st.divider()
    st.caption(
        "⚠️ If you change a key mid-session, restart the app — the LLM "
        "client is built once at import time from the environment."
    )

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

keys_ready = bool(gemini_key and tavily_key)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔎 Multi-Agent Research System")
st.caption(
    "Search agent → Reader agent → Writer chain → Critic chain, powered by "
    "Gemini + Tavily + LangChain."
)

if not keys_ready:
    st.info("Enter your Gemini and Tavily API keys in the sidebar to get started.")
    st.stop()

# Import only once keys are in the environment (agents.py reads them at
# import time). Cached so we don't rebuild the agents on every rerun.
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from pipeline import run_research_pipeline
    return run_research_pipeline

try:
    run_research_pipeline = _load_pipeline()
except Exception as e:
    st.error(f"Failed to load the pipeline. Check your dependencies and keys.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
topic = st.text_input(
    "Research topic",
    placeholder="e.g. The impact of quantum computing on cybersecurity",
)
run_clicked = st.button("Run Research", type="primary", disabled=not topic.strip())

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked and topic.strip():
    progress = st.status("Running the research pipeline...", expanded=True)
    try:
        progress.write("🔍 Search agent is gathering sources...")
        t0 = time.time()
        result = run_research_pipeline(topic.strip())
        progress.write(f"✅ Done in {time.time() - t0:.1f}s")
        progress.update(label="Pipeline complete", state="complete", expanded=False)
        st.session_state.result = result
        st.session_state.topic = topic.strip()
    except Exception as e:
        progress.update(label="Pipeline failed", state="error")
        st.error(f"Something went wrong: {e}")
        st.session_state.result = None

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result
if result:
    st.divider()
    st.subheader(f"Results for: {st.session_state.get('topic', '')}")

    tab_report, tab_critic, tab_search, tab_scraped = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔍 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(result.get("report", "_No report generated._"))
        st.download_button(
            "⬇️ Download report (.md)",
            data=result.get("report", ""),
            file_name=f"research_report_{st.session_state.get('topic','topic').replace(' ', '_')}.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(result.get("feedback", "_No feedback generated._"))

    with tab_search:
        st.text(result.get("search_results", "_No search results._"))

    with tab_scraped:
        st.text(result.get("scraped_content", "_No scraped content._"))
else:
    st.caption("Enter a topic above and click **Run Research** to begin.")