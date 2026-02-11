"""
Manual-RAG Diagnostic Assistant
================================
AI-powered equipment diagnosis from technical manuals.
Runs 100% offline using local LLM (Ollama) + local embeddings + local vector DB.

Architecture:
  PDF Manuals → Document Processor → ChromaDB (per-equipment) → Ollama LLM → Diagnosis

Launch:
  streamlit run app.py
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doc_processor import process_pdf, get_processing_stats, DocumentChunk
from vector_store import VectorStore
from llm_engine import (
    check_ollama_status,
    get_available_models,
    generate_response,
    generate_response_full,
    ConversationMemory,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Manual-RAG Diagnostic Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Dark industrial theme */
    .main .block-container { max-width: 1200px; padding-top: 1rem; }

    .equipment-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .equipment-card:hover { transform: translateY(-2px); }

    .stat-box {
        background: #0f3460;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin: 0.3rem;
    }
    .stat-box h3 { margin: 0; font-size: 1.8rem; color: #e94560; }
    .stat-box p { margin: 0; font-size: 0.8rem; color: #a0a0a0; }

    .source-badge {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.75rem;
        color: #a0a0a0;
    }

    .status-ok { color: #00d26a; font-weight: bold; }
    .status-err { color: #e94560; font-weight: bold; }

    .how-to-step {
        background: #16213e;
        border-left: 3px solid #e94560;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    .chat-msg-user {
        background: #0f3460;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-msg-ai {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "vector_store": None,
        "active_equipment": None,
        "conversation_memory": ConversationMemory(),
        "chat_history": [],
        "processing_status": None,
        "selected_model": DEFAULT_MODEL,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "n_results": 5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Initialize vector store
    if st.session_state["vector_store"] is None:
        st.session_state["vector_store"] = VectorStore()

init_session_state()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_vs() -> VectorStore:
    return st.session_state["vector_store"]


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## Manual-RAG Diagnostic Assistant")
        st.markdown("*AI diagnosis from your technical manuals*")
        st.markdown("---")

        # --- System Status ---
        st.markdown("### System Status")
        ollama_status = check_ollama_status()

        if ollama_status["running"]:
            st.markdown(f'<span class="status-ok">OLLAMA: ONLINE</span>', unsafe_allow_html=True)
            models = ollama_status.get("models", [])
            if models:
                st.session_state["selected_model"] = st.selectbox(
                    "LLM Model",
                    options=models,
                    index=models.index(st.session_state["selected_model"])
                    if st.session_state["selected_model"] in models else 0,
                )
            else:
                st.warning("No models found. Pull a model:\n`ollama pull llama3.1:8b`")
        else:
            st.markdown(f'<span class="status-err">OLLAMA: OFFLINE</span>', unsafe_allow_html=True)
            st.error("Start Ollama server:\n```\nollama serve\n```")

        st.markdown("---")

        # --- Equipment Selector ---
        st.markdown("### Equipment")
        vs = get_vs()
        equipment_list = vs.list_equipment()

        if equipment_list:
            equip_options = {e.equipment_id: f"{e.name} ({e.chunk_count} chunks)" for e in equipment_list}
            selected = st.selectbox(
                "Select Equipment",
                options=list(equip_options.keys()),
                format_func=lambda x: equip_options[x],
                index=list(equip_options.keys()).index(st.session_state["active_equipment"])
                if st.session_state["active_equipment"] in equip_options else 0,
            )
            if selected != st.session_state["active_equipment"]:
                st.session_state["active_equipment"] = selected
                st.session_state["chat_history"] = []
                st.session_state["conversation_memory"].clear()
                st.rerun()
        else:
            st.info("No equipment registered yet.\nGo to **Equipment Manager** to add one.")

        st.markdown("---")

        # --- Settings ---
        with st.expander("Advanced Settings"):
            st.session_state["chunk_size"] = st.slider(
                "Chunk Size (chars)", 200, 2000, st.session_state["chunk_size"], 100
            )
            st.session_state["chunk_overlap"] = st.slider(
                "Chunk Overlap", 50, 500, st.session_state["chunk_overlap"], 50
            )
            st.session_state["n_results"] = st.slider(
                "Context Chunks (retrieval)", 1, 15, st.session_state["n_results"]
            )

        st.markdown("---")
        st.markdown(
            "**Navigation**\n"
            "- Diagnostic Chat\n"
            "- Equipment Manager\n"
            "- Upload Manuals\n"
            "- System Guide"
        )


render_sidebar()


# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------

tab_chat, tab_equipment, tab_upload, tab_guide = st.tabs([
    "Diagnostic Chat", "Equipment Manager", "Upload Manuals", "System Guide"
])


# ===================== TAB 1: DIAGNOSTIC CHAT =============================

with tab_chat:
    st.markdown("## Diagnostic Chat")

    active_eq = st.session_state["active_equipment"]
    vs = get_vs()

    if not active_eq:
        st.info("Select or create an equipment in the **Equipment Manager** tab first.")
    else:
        equip_info = vs.get_equipment(active_eq)
        if equip_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="stat-box">
                    <h3>{equip_info.name}</h3>
                    <p>Active Equipment</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stat-box">
                    <h3>{equip_info.chunk_count}</h3>
                    <p>Knowledge Chunks</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="stat-box">
                    <h3>{equip_info.manual_count}</h3>
                    <p>Manuals Loaded</p>
                </div>""", unsafe_allow_html=True)

        if equip_info and equip_info.chunk_count == 0:
            st.warning("No manuals uploaded for this equipment. Go to **Upload Manuals** tab.")
        else:
            # --- Chat History ---
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    if msg["role"] == "user":
                        st.markdown(f"""<div class="chat-msg-user">
                            <strong>You:</strong> {msg["content"]}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="chat-msg-ai">
                            {msg["content"]}
                        </div>""", unsafe_allow_html=True)
                        # Source badges
                        if msg.get("sources"):
                            badges = ""
                            for src in msg["sources"]:
                                badges += f'<span class="source-badge">{src["source_file"]} p.{src["page_number"]} [{src["chunk_type"]}]</span> '
                            st.markdown(badges, unsafe_allow_html=True)

            # --- Input ---
            user_input = st.chat_input(
                f"Ask about {equip_info.name if equip_info else 'equipment'}..."
            )

            if user_input:
                # Add user message
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": user_input,
                })

                # Retrieve context
                with st.spinner("Searching manuals..."):
                    results = vs.query(
                        active_eq,
                        user_input,
                        n_results=st.session_state["n_results"],
                    )

                # Generate response
                with st.spinner("Analyzing with local LLM..."):
                    try:
                        response_text = generate_response_full(
                            question=user_input,
                            retrieved_chunks=results,
                            model=st.session_state["selected_model"],
                            equipment_name=equip_info.name if equip_info else "",
                        )

                        sources = [
                            {
                                "source_file": r["source_file"],
                                "page_number": r["page_number"],
                                "chunk_type": r["chunk_type"],
                            }
                            for r in results
                        ]

                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                        })

                        # Update conversation memory
                        st.session_state["conversation_memory"].add_exchange(
                            user_input, response_text, sources
                        )

                    except Exception as e:
                        error_msg = str(e)
                        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                            st.error("Cannot connect to Ollama. Make sure it's running:\n```\nollama serve\n```")
                        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                            st.error(f"Model not found. Pull it first:\n```\nollama pull {st.session_state['selected_model']}\n```")
                        else:
                            st.error(f"LLM Error: {error_msg}")

                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": f"*Error: {error_msg}*",
                        })

                st.rerun()

            # Clear chat
            if st.session_state["chat_history"]:
                if st.button("Clear Chat History"):
                    st.session_state["chat_history"] = []
                    st.session_state["conversation_memory"].clear()
                    st.rerun()


# ===================== TAB 2: EQUIPMENT MANAGER ===========================

with tab_equipment:
    st.markdown("## Equipment Manager")
    st.markdown("Each equipment has its own isolated knowledge base. "
                "Manuals for different equipment are never mixed.")

    vs = get_vs()

    # --- Register new equipment ---
    st.markdown("### Register New Equipment")
    with st.form("register_equipment"):
        col1, col2 = st.columns(2)
        with col1:
            eq_id = st.text_input(
                "Equipment ID",
                placeholder="e.g., main_engine_01",
                help="Unique identifier — no spaces, use underscores"
            )
        with col2:
            eq_name = st.text_input(
                "Equipment Name",
                placeholder="e.g., MAN B&W 6S50ME-C Main Engine"
            )
        eq_desc = st.text_area(
            "Description (optional)",
            placeholder="e.g., Main propulsion engine, 2-stroke, 6-cylinder...",
            height=80,
        )
        submitted = st.form_submit_button("Register Equipment")

        if submitted:
            if not eq_id or not eq_name:
                st.error("Equipment ID and Name are required.")
            elif eq_id in [e.equipment_id for e in vs.list_equipment()]:
                st.error(f"Equipment '{eq_id}' already exists.")
            else:
                vs.register_equipment(eq_id, eq_name, eq_desc)
                st.session_state["active_equipment"] = eq_id
                st.success(f"Registered: **{eq_name}**")
                st.rerun()

    # --- Existing equipment ---
    st.markdown("### Registered Equipment")
    equipment_list = vs.list_equipment()

    if not equipment_list:
        st.info("No equipment registered. Use the form above to add your first one.")
    else:
        for equip in equipment_list:
            with st.container():
                st.markdown(f"""<div class="equipment-card">
                    <h4>{equip.name}</h4>
                    <p><strong>ID:</strong> {equip.equipment_id}</p>
                    <p>{equip.description}</p>
                    <p><strong>Manuals:</strong> {equip.manual_count} | <strong>Knowledge Chunks:</strong> {equip.chunk_count}</p>
                </div>""", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    if st.button(f"Select", key=f"select_{equip.equipment_id}"):
                        st.session_state["active_equipment"] = equip.equipment_id
                        st.session_state["chat_history"] = []
                        st.session_state["conversation_memory"].clear()
                        st.rerun()
                with col2:
                    stats = vs.get_collection_stats(equip.equipment_id)
                    if stats:
                        st.caption(f"Chunks: {stats.get('total_chunks', 0)}")
                with col3:
                    if st.button(f"Delete", key=f"delete_{equip.equipment_id}", type="secondary"):
                        vs.delete_equipment(equip.equipment_id)
                        if st.session_state["active_equipment"] == equip.equipment_id:
                            st.session_state["active_equipment"] = None
                        st.rerun()


# ===================== TAB 3: UPLOAD MANUALS ==============================

with tab_upload:
    st.markdown("## Upload Manuals")

    vs = get_vs()
    equipment_list = vs.list_equipment()

    if not equipment_list:
        st.warning("Register equipment first in the **Equipment Manager** tab.")
    else:
        # Select target equipment
        target_eq = st.selectbox(
            "Upload to Equipment",
            options=[e.equipment_id for e in equipment_list],
            format_func=lambda x: next(
                (e.name for e in equipment_list if e.equipment_id == x), x
            ),
        )

        st.markdown("### Upload PDF Manuals")
        st.markdown(
            "Upload technical manuals (PDF format). The system will extract:\n"
            "- **Text** — All readable text content\n"
            "- **Tables** — Structured data (specs, clearances, tolerances)\n"
            "- **Images/Diagrams** — OCR extraction of text from diagrams\n"
        )

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF manuals for this equipment"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
            for f in uploaded_files:
                st.markdown(f"- {f.name} ({f.size / 1024 / 1024:.1f} MB)")

            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.session_state["chunk_size"]
                st.caption(f"Chunk size: {chunk_size} chars")
            with col2:
                chunk_overlap = st.session_state["chunk_overlap"]
                st.caption(f"Chunk overlap: {chunk_overlap} chars")

            if st.button("Process & Ingest Manuals", type="primary"):
                total_chunks = 0
                overall_progress = st.progress(0.0)
                status_text = st.empty()
                stats_container = st.container()

                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status_text.markdown(f"**Processing:** {uploaded_file.name} "
                                       f"({file_idx + 1}/{len(uploaded_files)})")

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        # Process PDF
                        def update_progress(stage, pct):
                            file_progress = (file_idx + pct) / len(uploaded_files)
                            overall_progress.progress(file_progress)
                            status_text.markdown(
                                f"**{uploaded_file.name}:** {stage} "
                                f"({file_idx + 1}/{len(uploaded_files)})"
                            )

                        chunks = process_pdf(
                            tmp_path,
                            target_eq,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            progress_callback=update_progress,
                        )

                        # Ingest into vector store
                        status_text.markdown(f"**{uploaded_file.name}:** Embedding & storing...")
                        added = vs.add_chunks(target_eq, chunks, uploaded_file.name)
                        total_chunks += added

                        # Show stats
                        proc_stats = get_processing_stats(chunks)
                        with stats_container:
                            st.markdown(f"**{uploaded_file.name}** — "
                                       f"{proc_stats['total_chunks']} chunks, "
                                       f"{proc_stats.get('pages_covered', 0)} pages, "
                                       f"Types: {proc_stats.get('chunks_by_type', {})}")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logger.error(f"Processing error: {e}", exc_info=True)

                    finally:
                        os.unlink(tmp_path)

                overall_progress.progress(1.0)
                status_text.markdown(f"**Done!** {total_chunks} total chunks ingested.")
                st.success(
                    f"Successfully processed {len(uploaded_files)} manual(s) → "
                    f"{total_chunks} knowledge chunks added to "
                    f"**{next((e.name for e in equipment_list if e.equipment_id == target_eq), target_eq)}**"
                )
                st.balloons()

        # --- Show existing data ---
        st.markdown("---")
        st.markdown("### Existing Knowledge Base")
        stats = vs.get_collection_stats(target_eq)
        if stats:
            st.json(stats)


# ===================== TAB 4: SYSTEM GUIDE ================================

with tab_guide:
    st.markdown("## System Guide")

    st.markdown("""
    ### What is Manual-RAG Diagnostic Assistant?

    A **100% offline AI diagnostic system** that reads your equipment's technical manuals
    and answers questions using a local Large Language Model (LLM). No internet required
    after initial setup. No data leaves your machine.

    **Key Features:**
    - Upload PDF manuals (text, tables, diagrams — all extracted)
    - Equipment-isolated knowledge bases (data never mixes)
    - Local AI reasoning with engineering depth
    - Source citations back to specific manual pages
    - Conversational follow-up questions

    ---

    ### Architecture
    """)

    st.code("""
    ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
    │  PDF Manual  │────>│  Doc Processor    │────>│  ChromaDB   │
    │  (upload)    │     │  • PyMuPDF (text) │     │  (per-equip │
    │              │     │  • pdfplumber     │     │   vectors)  │
    │              │     │    (tables)       │     └──────┬──────┘
    │              │     │  • Tesseract OCR  │            │
    │              │     │    (images)       │            │ query
    └─────────────┘     │  • Chunking       │            │
                        └──────────────────┘            v
    ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
    │  User Chat   │────>│  Retrieved chunks │────>│  Ollama LLM │
    │  (question)  │     │  (top-k similar)  │     │  (local AI) │
    └─────────────┘     └──────────────────┘     └──────┬──────┘
                                                        │
                                                        v
                                                 ┌─────────────┐
                                                 │  Diagnosis   │
                                                 │  + Sources   │
                                                 └─────────────┘
    """, language=None)

    st.markdown("""
    ---

    ### Step-by-Step Setup
    """)

    steps = [
        ("Step 1: Install System Dependencies", """
```bash
# Python 3.10+
python --version

# Tesseract OCR (for diagram text extraction)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```"""),
        ("Step 2: Install Ollama (Local LLM Server)", """
```bash
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Start the server:
ollama serve

# Pull a model (choose based on your RAM):
# 8GB RAM  → ollama pull phi3:3.8b
# 16GB RAM → ollama pull llama3.1:8b
# 32GB+ RAM → ollama pull llama3.1:70b
```"""),
        ("Step 3: Install Python Dependencies", """
```bash
pip install -r requirements.txt
```
First run will download the embedding model (~90MB) automatically."""),
        ("Step 4: Launch the Application", """
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`"""),
        ("Step 5: Register Your Equipment", """
Go to **Equipment Manager** tab:
1. Enter a unique ID (e.g., `main_engine_01`)
2. Enter the equipment name (e.g., `MAN B&W 6S50ME-C`)
3. Add a description
4. Click **Register Equipment**

Each equipment gets its own isolated knowledge base."""),
        ("Step 6: Upload Manuals", """
Go to **Upload Manuals** tab:
1. Select the target equipment
2. Upload PDF files (technical manuals, service bulletins, etc.)
3. Click **Process & Ingest Manuals**
4. Wait for processing (text + tables + OCR)

The system extracts everything: text, tables, diagrams."""),
        ("Step 7: Start Diagnosing", """
Go to **Diagnostic Chat** tab:
1. Select your equipment from the sidebar
2. Ask questions like:
   - *"What are the cylinder liner clearances?"*
   - *"Troubleshoot high exhaust temperature on cylinder 3"*
   - *"What is the procedure for fuel injector overhaul?"*
3. The AI responds with manual-based diagnosis + source citations"""),
    ]

    for title, content in steps:
        st.markdown(f"""<div class="how-to-step">
            <strong>{title}</strong>
        </div>""", unsafe_allow_html=True)
        st.markdown(content)

    st.markdown("""
    ---

    ### Hardware Requirements
    """)

    st.markdown("""
    | Component | Minimum | Recommended | Best |
    |-----------|---------|-------------|------|
    | **CPU** | 4 cores | 8+ cores | 12+ cores |
    | **RAM** | 8 GB | 16 GB | 32+ GB |
    | **Storage** | 20 GB free | 50 GB SSD | 100+ GB NVMe |
    | **GPU** | Not required | NVIDIA 8GB+ | NVIDIA 16GB+ |
    | **OS** | Linux / macOS / Windows | Ubuntu 22.04+ | Ubuntu 22.04+ |

    **LLM Model Selection by RAM:**

    | RAM | Model | Command | Quality |
    |-----|-------|---------|---------|
    | 8 GB | Phi-3 3.8B | `ollama pull phi3:3.8b` | Good |
    | 16 GB | Llama 3.1 8B | `ollama pull llama3.1:8b` | Very Good |
    | 32 GB | Llama 3.1 70B (Q4) | `ollama pull llama3.1:70b` | Excellent |
    | 48+ GB | Llama 3.1 70B (Q8) | `ollama pull llama3.1:70b-q8_0` | Best |

    **With NVIDIA GPU:**

    | VRAM | Model | Speed Boost |
    |------|-------|-------------|
    | 8 GB | Llama 3.1 8B | ~3x faster |
    | 16 GB | Llama 3.1 8B | ~5x faster |
    | 24 GB | Llama 3.1 70B (Q4) | ~3x faster |
    """)

    st.markdown("""
    ---

    ### Data Privacy & Security

    - **100% Offline** — No data leaves your machine after initial setup
    - **No Cloud APIs** — All AI processing runs locally via Ollama
    - **Equipment Isolation** — Each equipment's data is stored in a separate database collection
    - **Your Data, Your Control** — Delete any equipment and its data at any time
    - **No Telemetry** — ChromaDB telemetry is disabled by default

    ---

    ### Troubleshooting

    | Issue | Solution |
    |-------|----------|
    | "Cannot connect to Ollama" | Run `ollama serve` in a terminal |
    | "Model not found" | Run `ollama pull <model-name>` |
    | OCR not working | Install Tesseract: `sudo apt-get install tesseract-ocr` |
    | Slow responses | Use a smaller model or add a GPU |
    | Low quality answers | Upload more manuals, or use a larger model |
    | Out of memory | Use a smaller model (phi3:3.8b) or add more RAM |
    """)

    st.markdown("""
    ---

    ### Example Questions

    **Diagnostic Questions:**
    - "The main engine exhaust temperature on unit 3 is 40°C above the mean. What are the possible causes?"
    - "What is the procedure to check and adjust fuel injection timing?"
    - "Scavenge air temperature is rising — what should I check first?"

    **Information Lookup:**
    - "What are the recommended clearances for the main bearing?"
    - "What is the lube oil specification for the turbocharger?"
    - "List all safety interlocks for the fuel oil system"

    **Maintenance Procedures:**
    - "Step-by-step procedure for piston ring replacement"
    - "How to calibrate the cylinder pressure sensor?"
    - "What are the torque values for the cylinder head bolts?"
    """)
