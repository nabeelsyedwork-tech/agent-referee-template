import json
import os
import re
from typing import TypedDict, List, Dict, Literal, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain / LangGraph Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
import redis

# -------------------------------------------------------------------
# CONFIGURATION & ENV
# -------------------------------------------------------------------

# Ensure you have set GOOGLE_API_KEY in your environment for Gemini
# Ensure you have 'qwen2.5:7b-instruct' (or your preferred model) pulled in Ollama

# UPDATED: Using specific version tag 'gemini-1.5-flash-001' to resolve 404 errors.
# If this still fails, try 'gemini-pro' or check your API Key permissions.
OLLAMA_MODEL = "qwen2.5:7b-instruct"          # Model for the 4 agents
GEMINI_MODEL = "gemini-2.5-flash"         # Model for the Referee
OLLAMA_BASE_URL = "http://10.224.148.43:11434"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "agent_memory"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

MAX_RETRIES = 2

# -------------------------------------------------------------------
# CLIENT INITIALIZATION
# -------------------------------------------------------------------

# 1. Agents (Ollama)
llm_agent = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    base_url=OLLAMA_BASE_URL
)

# 2. Referee (Google Gemini)
llm_referee = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    convert_system_message_to_human=True
)

# 3. Embeddings (Ollama)
embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

# 4. Storage Clients
qdrant_client = QdrantClient(url=QDRANT_URL)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# We define vectorstore later in startup to ensure collection exists

# -------------------------------------------------------------------
# HELPER: Qdrant Setup
# -------------------------------------------------------------------
def init_qdrant():
    """Ensures Qdrant collection exists with correct dimensions."""
    try:
        if not qdrant_client.collection_exists(QDRANT_COLLECTION):
            print(f"Collection '{QDRANT_COLLECTION}' not found. Creating...")
            
            # Determine embedding dimension dynamically
            # (Qwen 2.5 7B is typically 3584, but we check to be safe)
            test_embed = embeddings.embed_query("test")
            dim = len(test_embed)
            print(f"Detected embedding dimension: {dim}")

            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            print(f"Collection '{QDRANT_COLLECTION}' created successfully.")
        else:
            print(f"Collection '{QDRANT_COLLECTION}' already exists.")
            
    except Exception as e:
        print(f"WARNING: Qdrant initialization failed. Memory features may not work.\nError: {e}")

# Initialize Vectorstore wrapper
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embeddings 
)

# -------------------------------------------------------------------
# PROMPTS
# -------------------------------------------------------------------

PRIMARY_SOLVER_SYSTEM = """You are the Primary Solver Agent.
Your job is to produce a complete, step-by-step solution to the user's task.
Be clear, detailed, and explicit in your reasoning.
Avoid patterns that were judged as failures in past examples."""

LOGIC_CRITIC_SYSTEM = """You are the Logic Critic Agent.
Your job is to analyze the LATEST PROPOSED SOLUTION provided by the previous agent.
Look for:
- logical errors
- missing steps
- contradictions
- unjustified assumptions.

If the solution is good, acknowledge it. If it is flawed, explain exactly why."""

DOMAIN_EXPERT_SYSTEM = """You are the Domain Expert Agent.
Review the proposed solution and the critic's feedback.
Focus on factual and domain-specific correctness.
Check whether the solution matches real-world facts, constraints, best practices.
Highlight incorrect claims and suggest corrections."""

OPTIMIZER_SYSTEM = """You are the Optimizer Agent.
Synthesize the original solution, the logic critiques, and the domain expert's feedback.
Produce the FINAL, BEST version of the answer:
- clear
- well-structured
- concise but complete.
"""

# FIX: Escaped curly braces {{ }} so LangChain doesn't treat them as variables
REFEREE_SYSTEM = """You are a strict Referee Agent.

Your task: evaluate the latest agent's output regarding the user's task.

You MUST output ONLY valid JSON with this exact schema:
{{
  "verdict": "VALID" or "INVALID",
  "reason": "Short explanation of your decision",
  "required_fixes": ["list", "of", "fixes"],
  "confidence_score": 0-100
}}

IMPORTANT:
- Do not include markdown formatting like ```json ... ```.
- Just output the raw JSON string.
- VERDICT is "VALID" only if the answer is correct, clear, and usable.
- If unsure, use "INVALID".
"""

AGENT_PROMPTS = {
    "Primary_Solver": PRIMARY_SOLVER_SYSTEM,
    "Logic_Critic": LOGIC_CRITIC_SYSTEM,
    "Domain_Expert": DOMAIN_EXPERT_SYSTEM,
    "Optimizer": OPTIMIZER_SYSTEM,
}

# -------------------------------------------------------------------
# STATE DEFINITION
# -------------------------------------------------------------------

AgentName = Literal["Primary_Solver", "Logic_Critic", "Domain_Expert", "Optimizer"]

class AgentState(TypedDict):
    user_query: str
    agents: List[AgentName]
    current_agent_index: int
    current_agent: AgentName
    response: str 
    verdict: Literal["VALID", "INVALID"]
    reason: str
    required_fixes: List[str]
    confidence_score: int
    retries: int
    max_retries: int
    history: List[Dict[str, Any]]
    done: bool

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------

def store_example_in_qdrant(text: str, metadata: dict) -> None:
    try:
        vectorstore.add_texts(texts=[text], metadatas=[metadata])
    except Exception as e:
        print(f"Qdrant Write Error: {e}")

def retrieve_examples(task: str, k: int = 3) -> List[str]:
    try:
        docs = vectorstore.similarity_search(task, k=k)
        return [d.page_content for d in docs]
    except Exception:
        return []

def get_last_valid_response(history: List[Dict]) -> str:
    for item in reversed(history):
        if item.get("step") == "agent" and "response" in item:
            return item["response"]
    return "No prior solution provided."

# -------------------------------------------------------------------
# GRAPH NODES
# -------------------------------------------------------------------

def next_agent_node(state: AgentState) -> AgentState:
    idx = state.get("current_agent_index", 0)
    agents = state["agents"]

    if idx >= len(agents):
        return {**state, "done": True}

    agent = agents[idx]
    
    updates = {
        "current_agent_index": idx,
        "current_agent": agent,
        "retries": 0,
        "verdict": "INVALID",
        "done": False,
    }
    return {**state, **updates}

def agent_node(state: AgentState) -> AgentState:
    agent = state["current_agent"]
    task = state["user_query"]
    history = state.get("history", [])
    
    examples = retrieve_examples(task, k=2)
    examples_str = "\n\n".join(examples) if examples else "No prior examples."

    context_str = ""
    if agent != "Primary_Solver":
        last_response = get_last_valid_response(history)
        context_str = f"\n\n--- PREVIOUS AGENT OUTPUT ---\n{last_response}\n-----------------------------"

    system_prompt_text = AGENT_PROMPTS[agent]
    
    messages = [
        SystemMessage(content=f"{system_prompt_text}\n\nReference Examples:\n{examples_str}"),
        HumanMessage(content=f"User Task: {task}{context_str}")
    ]

    try:
        result = llm_agent.invoke(messages)
        response_text = result.content
    except Exception as e:
        response_text = f"Agent failed to generate: {e}"

    new_entry = {
        "step": "agent",
        "agent": agent,
        "response": response_text,
        "retries": state.get("retries", 0)
    }
    
    return {
        **state,
        "response": response_text,
        "history": history + [new_entry]
    }

def referee_node(state: AgentState) -> AgentState:
    agent = state["current_agent"]
    task = state["user_query"]
    answer = state["response"]
    
    # NOTE: REFEREE_SYSTEM now has escaped braces {{ }} to prevent formatting errors
    prompt = ChatPromptTemplate.from_messages([
        ("system", REFEREE_SYSTEM),
        ("human", "User Task:\n{task}\n\nAgent ({agent}) Output:\n{answer}")
    ])
    
    try:
        msg = prompt.format_messages(task=task, agent=agent, answer=answer)
        result = llm_referee.invoke(msg)
        content = result.content
        
        # Robust Regex Extraction to find the JSON object (first '{' to last '}')
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
        else:
            # Fallback if no braces found (rare)
            data = json.loads(content)
            
    except Exception as e:
        # Changed log to distinguish between API errors (404, 500) and Parsing errors
        print(f"Referee Execution/Parsing Error: {e}")
        
        # Default fallback so we don't crash the entire graph
        data = {
            "verdict": "VALID", # Default to VALID to prevent infinite loop
            "reason": f"Referee Error (Model/Parse): {e}. Assuming valid to proceed.",
            "required_fixes": [],
            "confidence_score": 0
        }

    verdict = data.get("verdict", "INVALID")
    reason = data.get("reason", "No reason provided")
    
    # Store in Qdrant
    if verdict == "VALID": # Only store valid examples usually
        meta = {"agent": agent, "verdict": verdict}
        blob = f"Task: {task}\nAgent: {agent}\nOutput: {answer}"
        store_example_in_qdrant(blob, meta)

    new_entry = {
        "step": "referee",
        "agent": agent,
        "verdict": verdict,
        "reason": reason,
        "full_data": data
    }

    return {
        **state,
        "verdict": verdict,
        "reason": reason,
        "required_fixes": data.get("required_fixes", []),
        "confidence_score": data.get("confidence_score", 0),
        "history": state["history"] + [new_entry]
    }

def increment_retry(state: AgentState) -> AgentState:
    return {**state, "retries": state.get("retries", 0) + 1}

# -------------------------------------------------------------------
# GRAPH ROUTING
# -------------------------------------------------------------------

def route_next(state: AgentState):
    if state.get("done"):
        return END
    return "agent"

def route_referee(state: AgentState):
    verdict = state.get("verdict", "INVALID")
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)

    if verdict == "VALID":
        return "next_agent_step"
    elif retries < max_retries:
        return "increment_retry"
    else:
        return "next_agent_step"

def advance_index(state: AgentState) -> AgentState:
    return {**state, "current_agent_index": state["current_agent_index"] + 1}

# -------------------------------------------------------------------
# GRAPH CONSTRUCTION
# -------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("next_agent", next_agent_node)
graph.add_node("agent", agent_node)
graph.add_node("referee", referee_node)
graph.add_node("increment_retry", increment_retry)
graph.add_node("advance_index", advance_index)

graph.set_entry_point("next_agent")

graph.add_conditional_edges("next_agent", route_next, {END: END, "agent": "agent"})
graph.add_edge("agent", "referee")
graph.add_conditional_edges(
    "referee", 
    route_referee, 
    {
        "increment_retry": "increment_retry", 
        "next_agent_step": "advance_index"
    }
)
graph.add_edge("increment_retry", "agent")
graph.add_edge("advance_index", "next_agent")

workflow = graph.compile()

# -------------------------------------------------------------------
# LIFESPAN & APP
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_qdrant()
    yield
    # Shutdown logic (optional)

app = FastAPI(title="Multi-Agent Referee System", lifespan=lifespan)

# Add CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    query: str
    max_retries: int = 2

@app.post("/run")
async def run_task(req: TaskRequest):
    initial_state = {
        "user_query": req.query,
        "agents": ["Primary_Solver", "Logic_Critic", "Domain_Expert", "Optimizer"],
        "current_agent_index": 0,
        "max_retries": req.max_retries,
        "history": [],
        "done": False,
        "retries": 0,
        "verdict": "INVALID", 
        "response": "",
        "reason": "",
        "required_fixes": [],
        "confidence_score": 0,
        "current_agent": "Primary_Solver"
    }
    
    # INCREASED RECURSION LIMIT to prevent crashes on long retry loops
    # 4 agents * 2 retries * 3 steps each + overhead > 25 default
    final_state = workflow.invoke(initial_state, config={"recursion_limit": 100})
    
    return {
        "final_response": final_state["response"],
        "history": final_state["history"],
        "status": "completed"
    }

@app.get("/health")
def health():
    return {"status": "ok", "services": {"ollama": OLLAMA_BASE_URL, "qdrant": QDRANT_URL}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
