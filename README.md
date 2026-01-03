<div align="center">

<h1>🧠 Agent Referee System</h1>

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
A reusable, correctness-first multi-agent template for building niche, high-stakes AI systems with deterministic validation.
</em></p>

</div>


---

## Overview

The **Agent Referee System** is a **multi-agent AI framework** designed to produce **high-quality, validated responses** instead of single-pass LLM outputs.

Rather than trusting one model, the system:

* Breaks reasoning into **specialized agents**
* Evaluates every step using an **independent referee**
* Retries or rejects flawed reasoning
* Stores only **approved answers** for future reuse

The result is a system optimized for **correctness, transparency, and reliability**.

---

## Who This Is For

This project is designed as a **base template** for building specialized AI systems where correctness, traceability, and validation matter.

Example use cases include:

- Domain-specific research assistants
- Legal or compliance reasoning systems
- Financial analysis pipelines
- Scientific or technical QA engines
- Internal tools requiring auditable AI decisions

Rather than being a single-purpose app, this repository provides a **foundational architecture** that can be extended, customized, and deployed across different domains.

---
## Why This Exists

Large language models are fluent — but fluency is not correctness.

For tasks involving:

* Complex reasoning
* Technical explanations
* High-confidence outputs
* Auditable decision paths

…a single unchecked response is often not enough.

This project explores a **review-driven AI architecture** that can be reused as a foundation for multiple niche applications, rather than a single monolithic chatbot:

> propose → critique → verify → refine → approve

---

## How It Works

The Agent Referee System executes a **controlled, multi-step reasoning workflow** where **generation and validation are explicitly separated**. Each stage is orchestrated by LangGraph and gated by a deterministic referee.

### Step-by-Step Flow

1. **User Query (Frontend)**
   The user submits a query via the HTML/CSS/JS frontend.

2. **FastAPI Entry Point**
   The query is sent to `POST /run`, which initializes the workflow and hands control to the LangGraph controller.

3. **LangGraph Controller (Orchestration Layer)**
   The controller:

   * Selects the current agent (Primary, Critic, Expert, Optimizer)
   * Tracks the agent index
   * Manages retry counts
   * Persists execution state via Redis

4. **Context Retrieval (Memory Layer)**

   * **Redis** provides short-term state and conversation history
   * **Qdrant** retrieves previously **referee-approved examples** to enrich agent context

5. **Agent Execution (Ollama Models)**
   The selected agent:

   * Receives the user query + retrieved context
   * Performs its single responsibility
   * Produces a **proposed response**

6. **Referee Validation (Gemini)**
   The proposed response is evaluated by the **Gemini Referee**, which:

   * Enforces a strict JSON-only schema
   * Returns `VALID` or `INVALID`
   * Provides feedback and a confidence score

7. **Routing Logic (Decision Gate)**
   Based on the referee verdict:

   * **VALID** → advance to the next agent
   * **INVALID** → retry the current agent (if retries remain)

8. **Retry Handling (If Needed)**
   If invalid and retries are available:

   * Retry count is incremented
   * The agent is re-executed with referee feedback

9. **Finalization**
   When all agents complete *or* retry limits are reached:

   * The final validated response is returned by FastAPI
   * Approved outputs are stored in Qdrant for future reuse

10. **Frontend Display**
    The frontend renders:

    * The final answer
    * The full agent and referee audit trail

---

### Key Design Principle

> **No response is trusted by default.**
> Every agent output must pass an independent validation step before progressing.

This design prioritizes **correctness, transparency, and reproducibility** over single-pass fluency.


---

## System Architecture

<div align="center">

<img
  src="assets/agent-referee-architecture.jpg"
  alt="Agent Referee System Architecture Diagram"
  width="900"
/>

</div>

---

## Agent Roles

| Agent                | Responsibility                                |
| -------------------- | --------------------------------------------- |
| **Primary Solver**   | Generates a complete, step-by-step solution   |
| **Logic Critic**     | Finds logical errors, gaps, or contradictions |
| **Domain Expert**    | Verifies factual and domain correctness       |
| **Optimizer**        | Produces the final polished response          |
| **Referee (Gemini)** | Approves or rejects each step                 |

---

## Gemini Referee (Validation Layer)

Validation is performed by **Google Gemini** using a strict JSON-only schema:

```json
{
  "verdict": "VALID | INVALID",
  "reason": "Why this decision was made",
  "required_fixes": ["specific issues"],
  "confidence_score": 0-100
}
```

**Key properties**

* Deterministic (temperature = 0)
* Schema-enforced output
* Prevents silent hallucinations
* Controls retry behavior

Only **VALID** outputs progress or get stored.

---

## Memory & Learning (Qdrant)

Validated responses are embedded and stored in **Qdrant**.

* Enables retrieval-augmented reasoning (RAG)
* Injects prior *approved* examples into future prompts
* Improves consistency over time
* Fails gracefully if storage is unavailable

Only **referee-approved outputs** are persisted.

---

## Frontend

A lightweight, client-side UI built with **Tailwind CSS**.

**Features**

* Chat interface
* Expandable agent workflow trace
* Markdown rendering + syntax highlighting
* Visual indicators for agent activity
* Mobile-friendly layout

Users can inspect *exactly* how an answer was produced.

---

## Tech Stack

### Backend

* **FastAPI**
* **LangChain + LangGraph**
* **Ollama (Qwen 2.5)** — agent models
* **Google Gemini** — referee
* **Qdrant** — vector memory
* **Redis** — caching / queue (extensible)

### Frontend

* HTML + Tailwind CSS
* Marked.js
* Highlight.js

---

## Project Structure

```
/
├── index.html   # Frontend UI
└── main.py      # FastAPI backend + agent workflow
```

---

## Quick Start

### Prerequisites

* Python 3.10+
* Docker
* Ollama
* Google Gemini API key

---

### Install Dependencies

```bash
pip install fastapi uvicorn langchain langgraph \
langchain-ollama langchain-google-genai \
langchain-qdrant qdrant-client redis
```

---

### Start Infrastructure

```bash
docker run -p 6333:6333 qdrant/qdrant
docker run -p 6379:6379 redis
```

---

### Pull Ollama Model

```bash
ollama pull qwen2.5:7b-instruct
```

---

### Set Gemini API Key

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

---

### Run Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### Run Frontend

Open `index.html` in your browser.

---

## Usage

**POST** `/run`

```json
{
  "query": "Explain the significance of the Riemann Hypothesis",
  "max_retries": 2
}
```

**Response**

```json
{
  "final_response": "...",
  "history": [...],
  "status": "completed"
}
```

The `history` field contains a **full audit trail** of agent outputs and referee decisions.

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

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes clearly
4. Open a pull request

For large changes, please open an issue first.

---

## Support

* Bug reports: GitHub Issues
* Feature requests: GitHub Issues
* Questions: Discussions (if enabled)

---

## Roadmap

* [ ] Redis response caching
* [ ] Improved memory ranking
* [ ] WebSocket streaming
* [ ] Multi-user sessions
* [ ] Authentication & quotas for referee usage

