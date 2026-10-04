# jev-bench

A/B benchmark of an agentic document-retrieval workflow — the same LangGraph graph run with two
interchangeable decision backends, Google Gemini and Jev (TypeSafe AI), to measure how much latency
and cost is recoverable by moving classification-style decisions off an autoregressive LLM.

Status: under construction. See `docs/METHODOLOGY.md` (forthcoming) for the full measurement design.
