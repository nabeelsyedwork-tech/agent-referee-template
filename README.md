<div align="center">

<h1>Agent Referee System</h1>

<!-- Animated subtitle -->
<p>
<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=18&duration=3000&pause=800&color=22C55E&center=true&vCenter=true&width=520&lines=Multi-Agent+Reasoning+with+Deterministic+Validation;Generate+%E2%86%92+Critique+%E2%86%92+Verify+%E2%86%92+Approve;Correctness+over+Confidence" alt="Animated description" />
</p>

<!-- Vibrant badges -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Production--Ready-05998B?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestrated-7C3AED?style=for-the-badge&logo=graphql&logoColor=white" />
  <img src="https://img.shields.io/badge/Referee-Google%20Gemini-EA4335?style=for-the-badge&logo=google&logoColor=white" />
</p>

<p><em>
A deterministic, multi-agent validation framework for building high-stakes AI systems.
</em></p>

</div>

---

## System Overview

The **Agent Referee System** is a production-ready architecture designed to solve the "correctness problem" in Large Language Models (LLMs).

Most LLM implementations optimize for speed and fluency. This system optimizes for **deterministic validity**. It achieves this by orchestrating a team of specialized micro-agents—each with a distinct responsibility (drafting, critiquing, verifying)—subjecting every output to a strict, schema-enforced **Referee Agent**.

> **The Core Concept**: No agent output is trusted by default. Every reasoning step must be explicitly validated by an independent arbiter before state progression.

---

## Target Use Cases

Designed as a foundational template for high-stakes domains:
*   **Legal & Compliance**: Auditable reasoning pipelines where every claim must be cited.
*   **Scientific Research**: Fact-checking assistants that verify hypotheses against strict axioms.
*   **Financial Analysis**: Multi-step reasoning providers that self-correct calculation errors.

---

## Architecture

<div align="center">
  <img src="assets/agent-referee-architecture.jpg" alt="System Architecture" width="800" />
</div>

Instead of a single monolithic prompt, the system decomposes complex reasoning into a **LangGraph** workflow:

### Agent Roles

| Role | Technical Responsibility |
| :--- | :--- |
| **Generator (Writer)** | Synthesizes initial solutions based on retrieved context (RAG). |
| **Critic (Logic)** | Scans for logical fallacies and internal contradictions using chain-of-thought analysis. |
| **Expert (Domain)** | Validates factual accuracy against strict domain constraints. |
| **Referee (Gemini)** | A temperature-0 evaluator that enforces JSON schemas and gates state transitions (Pass/Reject). |

### The Validation Loop

1.  **Generate**: Agent produces a candidate response.
2.  **Evaluate**: Referee grades the response (`VALID` / `INVALID`) with structured feedback.
3.  **Loop**: If `INVALID`, the feedback is injected back into the context, and the agent retries (up to `max_retries`).
4.  **Commit**: Only `VALID` states are persisted to the vector memory.

---

## Technology Stack

Built for extensibility and scale:

*   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful multi-agent workflows)
*   **Validation**: Google Gemini (via `langchain-google-genai`)
*   **Inference**: Ollama (Local LLMs like Qwen 2.5, Llama 3)
*   **Memory**: [Qdrant](https://qdrant.tech/) (Vector store for approved reasoning paths)
*   **API**: FastAPI (Async, production-grade endpoints)
*   **Frontend**: Vanilla JS + Tailwind CSS (Lightweight visualization)

---

## Project Structure

```text
/
├── main.py           # FastAPI entry point & LangGraph orchestration
├── agents/           # Prompts and logic for specific agents (Writer, Critic, etc.)
├── core/             # Shared utilities (DB connections, text processing)
├── assets/           # Diagrams and static images
└── index.html        # Client-side visualization UI
```

---

## API Usage

**Endpoint**: `POST /run`

**Request**:
```json
{
  "query": "Explain the significance of the Riemann Hypothesis",
  "max_retries": 2
}
```

**Response**:
```json
{
  "final_response": "The Riemann Hypothesis...",
  "history": [
    { "agent": "Writer", "output": "...", "referee_verdict": "INVALID", "critique": "..." },
    { "agent": "Writer", "output": "...", "referee_verdict": "VALID" }
  ],
  "status": "completed"
}
```

---

## Quick Start

Prerequisites: `Docker`, `Python 3.10+`, `Ollama`.

### 1. Clone & Install
```bash
git clone https://github.com/your-username/agent-referee.git
cd agent-referee
pip install -r requirements.txt
```

### 2. Launch Infrastructure
Spin up the vector database and cache:
```bash
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis
```

### 3. Initialize Models
Pull the agent model locally:
```bash
ollama pull qwen2.5:7b-instruct
```

### 4. Run the Stack
Set your API key for the Referee (Gemini) and start the server:
```bash
export GOOGLE_API_KEY="your_api_key_here"
uvicorn main:app --reload
```

Access the UI at `http://localhost:8000/index.html` to visualize the agent traces in real-time.

---

## Health Check

**GET** `/health`

Returns backend and service status.

---

## Project Status

This project is **actively developed** and suitable for:

* Research experiments
* Advanced prompting pipelines
* Reasoning transparency demos
* Multi-agent system exploration

---

## Roadmap & Contributing

We are actively exploring:
- [ ] **Recursive RAG**: Indexing the *process* of reasoning, not just the result.
- [ ] **Token-level confidence**: Integrating logprobs for finer-grained refusal.
- [ ] **Streaming**: WebSocket implementation for real-time thought rendering.

Contributions specifically targeting **latency optimization** and **prompt hardening** are highly encouraged.
