# rag-agent-langgraph

A financial analysis agent built with LangGraph that combines real-time market data with SEC filing retrieval. The agent decides which tools to call, reasons across multiple data sources, and knows when to stop.

---

## What this does

The agent answers financial questions by orchestrating three types of tools:

- **Real-time data** — current stock price, market cap, volume via yfinance
- **Financial metrics** — P/E ratio, EPS, beta, 52-week range via yfinance
- **Document retrieval** — Apple's 10-K SEC filing indexed in FAISS, searched via semantic similarity

For a question like "Is Apple trading closer to its 52-week high or low, and what does their 10-K say about growth risk?" the agent calls the right tools in the right order, combines the results, and produces a single coherent answer.

---

## Architecture

```
User question
      ↓
 [LLM node] → should I call a tool?
      ↓ yes                ↓ no
 [Tool node]             [END]
      ↓
 [LLM node] → do I have enough to answer?
      ↓ no                 ↓ yes
 [Tool node]             [END]
```

Built on LangGraph's stateful graph — each node reads from and writes to a shared state object. The LLM decides when to call tools and when to stop by whether it populates `tool_calls` in its response.

---

## Stack

- **LangGraph** — agent orchestration and graph execution
- **Groq** — LLM inference (`openai/gpt-oss-20b`)
- **yfinance** — real-time financial data
- **FAISS** — local vector store for SEC filing retrieval
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` for document and query embedding
- **LangSmith** — tracing and run inspection

---

## Project structure

```
file_agent.py           # LangGraph agent + yfinance tools
file_rag.py    # EDGAR fetch, chunking, FAISS indexing, retriever tool
agent_with_rag.py  # combines agent + RAG into one pipeline
file_eval.py            # evaluation harness — keyword, tool, and latency scoring
```

---

## Eval harness

`file_eval.py` runs a small gold dataset through the agent and scores each response across three dimensions:

- **Keyword score** — did the response contain the expected information?
- **Tool score** — did the agent call the right tools, and only those tools?
- **Latency score** — did it respond within an acceptable time?

Sample results:

```
ID     Overall   Keywords    Tools    Latency
──────────────────────────────────────────────
q1      0.90       0.71      1.00      2.31s
q2      0.93       0.80      1.00      1.54s
q3      1.00       1.00      1.00      3.17s
q4      0.67       1.00      1.00     26.42s
q5      0.67       1.00      1.00     21.85s
q6      1.00       1.00      1.00      0.52s
q7      0.67       1.00      1.00     13.41s
──────────────────────────────────────────────
AVG     0.83
```

Tool selection is reliable across all questions including out-of-scope ones (q6 — weather question, zero tools called). Latency degrades on complex multi-step reasoning questions.

---

## Setup

```bash
git clone https://github.com/amankumar17-hub/rag-agent-langgraph
cd rag-agent-langgraph
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=financial-agent
```

Get a free Groq key at console.groq.com. LangSmith key at smith.langchain.com.

---

## Run

```bash
# run the agent directly
python file_agent.py

# run with RAG pipeline
python agent_with_rag.py

# run eval harness
python file_eval.py
```