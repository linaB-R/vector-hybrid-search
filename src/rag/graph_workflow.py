"""
LangGraph workflow wiring for multimodal RAG.

Graph nodes (each writes inputs/outputs into a shared state dict):
  1) router           -> decides: image_search | text_vector | hybrid | sql | mixed
  2) embed_query      -> compute query embeddings (text/image) when needed
  3) retrieve         -> call vector_store OR hybrid_retriever OR sql_tools
  4) merge_rerank     -> optional LLM rerank or rubric-based filtering
  5) compose_answer   -> call OpenAI Responses API; build final JSON payload
  6) trace_persist    -> persist artifacts (ids, scores, sql, images, params)

State shape suggestion:
  {
    "input": {"text": str, "image": Optional[bytes], "mode": str, "k": int, ...},
    "intent": str,
    "embeddings": {...},
    "candidates": [{"id":..., "score":..., ...}, ...],
    "sql": {"query": str, "result": {...}},
    "final": {"answer": str, "citations": [...], "images": [...], "tables": [...]}
  }

Team TODO:
- Instantiate & inject VectorStoreAdapter, HybridRetriever, SQLTool.
- Implement each node function and edges between them.
"""

from typing import Dict, Any
# from langgraph.* import ...   # TODO: import the specific primitives your team prefers

class RAGGraph:
    def __init__(self, vector_store, hybrid_retriever, sql_tool):
        self.vector_store = vector_store
        self.hybrid = hybrid_retriever
        self.sql_tool = sql_tool
        # TODO: build the graph with nodes & edges

    # --- Nodes (called by the graph engine) ---

    def node_router(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: classify intent and set state['intent']."""
        raise NotImplementedError

    def node_embed_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: produce query embeddings as needed; store under state['embeddings']."""
        raise NotImplementedError

    def node_retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: call vector_store / hybrid / sql_tool based on intent; put into state['candidates'] or state['sql']."""
        raise NotImplementedError

    def node_merge_rerank(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: optional second-pass LLM rerank or heuristic filtering."""
        raise NotImplementedError

    def node_compose_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO:
          - Call OpenAI Responses API with tool-calling enabled
          - Provide retrieved context (snippets, images, sql result) as input
          - Require a structured JSON output: final_answer, cited_items, sql_used, images_used
        """
        raise NotImplementedError

    def node_trace_persist(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: write compact JSON trace (params, timings, ids, sql, scores) for audit/demo."""
        raise NotImplementedError

    def compile(self):
        """TODO: return a runnable graph object (per LangGraph patterns)."""
        raise NotImplementedError
