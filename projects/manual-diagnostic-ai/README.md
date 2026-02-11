# Manual-RAG Diagnostic Assistant

**AI-powered equipment diagnosis from your technical manuals — runs 100% offline.**

Upload PDF manuals (text, tables, diagrams). Ask diagnostic questions. Get engineering-grade answers with source citations. No internet required. No data leaves your machine.

---

## Architecture

```
PDF Manuals → Doc Processor → ChromaDB (per-equipment) → Ollama LLM → Diagnosis
                 │                    │                       │
                 ├─ PyMuPDF (text)    ├─ Equipment A DB       ├─ Llama 3.1 8B
                 ├─ pdfplumber (tables)├─ Equipment B DB       ├─ Phi-3 3.8B
                 ├─ Tesseract (OCR)   └─ Equipment C DB       └─ or any Ollama model
                 └─ Sentence chunking
```

**Key Design Decisions:**
- **Equipment isolation** — each piece of machinery gets its own ChromaDB collection. Data from a main engine manual never mixes with generator manual data.
- **Multi-modal extraction** — PDFs aren't just text. Tables contain critical specs (clearances, tolerances). Diagrams contain flow paths and schematics. We extract all three.
- **Local-first** — after initial `pip install` and model download, the entire system runs offline. Built for ship engine rooms, remote sites, secure facilities.
- **Engineering reasoning** — the LLM prompt is tuned for diagnostic depth: symptom analysis → causal chain → manual-referenced procedures → corrective actions.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **PDF Text** | PyMuPDF | Fastest Python PDF parser, handles complex layouts |
| **Tables** | pdfplumber | Best table extraction accuracy for technical docs |
| **OCR** | Tesseract | Proven, offline, handles diagram text |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim, fast CPU inference, ~22M params |
| **Vector DB** | ChromaDB | Embedded, persistent, zero-config |
| **LLM** | Ollama (Llama 3.1 / Phi-3) | Local inference, no API keys, GPU optional |
| **UI** | Streamlit | Rapid deployment, no frontend build step |

---

## Quick Start

### 1. System Dependencies

```bash
# Tesseract OCR
sudo apt-get install tesseract-ocr    # Ubuntu/Debian
brew install tesseract                  # macOS
```

### 2. Ollama (Local LLM)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve                             # Start server
ollama pull llama3.1:8b                  # Pull model (16GB RAM)
# OR
ollama pull phi3:3.8b                    # Lighter model (8GB RAM)
```

### 3. Python Setup

```bash
pip install -r requirements.txt
```

### 4. Launch

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Usage

### Register Equipment
**Equipment Manager** → Enter ID, name, description → **Register**

Each equipment gets isolated storage. A ship might have:
- `main_engine_01` — MAN B&W 6S50ME-C
- `generator_01` — Wartsila 6L20
- `boiler_01` — Aalborg OL Boiler

### Upload Manuals
**Upload Manuals** → Select equipment → Upload PDFs → **Process & Ingest**

The processor extracts:
- Text content (instructions, procedures, descriptions)
- Tables (clearances, specs, tolerances, torque values)
- Diagram/image text via OCR (flow diagrams, schematics)

### Diagnostic Chat
**Diagnostic Chat** → Select equipment → Ask questions

Example questions:
- *"Troubleshoot high exhaust temperature on cylinder 3"*
- *"What are the main bearing clearances?"*
- *"Step-by-step fuel injector overhaul procedure"*
- *"List safety interlocks for the fuel oil system"*

Every answer cites the source manual and page number.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 20 GB free | 50 GB SSD |
| **GPU** | Not required | NVIDIA 8GB+ VRAM (3x speed) |

### Model Selection by RAM

| RAM | Model | Command | Quality |
|-----|-------|---------|---------|
| 8 GB | Phi-3 3.8B | `ollama pull phi3:3.8b` | Good |
| 16 GB | Llama 3.1 8B | `ollama pull llama3.1:8b` | Very Good |
| 32+ GB | Llama 3.1 70B | `ollama pull llama3.1:70b` | Excellent |

---

## Project Structure

```
manual-diagnostic-ai/
├── app.py              # Streamlit UI (4 tabs: Chat, Equipment, Upload, Guide)
├── doc_processor.py    # PDF extraction pipeline (text + tables + OCR + chunking)
├── vector_store.py     # ChromaDB with equipment-isolated collections
├── llm_engine.py       # Ollama integration + diagnostic prompt engineering
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
├── .gitignore
├── LICENSE             # MIT
└── README.md
```

---

## Data Privacy

- **100% offline** after setup — no network calls during operation
- **No cloud APIs** — all AI runs locally via Ollama
- **Equipment isolation** — separate ChromaDB collections per equipment
- **No telemetry** — ChromaDB telemetry disabled
- **Delete anytime** — remove any equipment and all its indexed data

---

## Supported Manual Formats

| Format | Text | Tables | Images/Diagrams |
|--------|------|--------|-----------------|
| Digital PDF | Full extraction | Full extraction | OCR extraction |
| Scanned PDF | OCR extraction | OCR + table detect | OCR extraction |
| Image-heavy PDF | Via OCR | Via OCR | Via OCR |

**Best results with:** Digital PDFs (not scanned), reasonable resolution, text-selectable content.

---

## License

MIT — use it, modify it, deploy it. Built for engineers who need answers from their manuals, not from the internet.
