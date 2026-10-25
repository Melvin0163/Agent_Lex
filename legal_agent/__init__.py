"""An AI-Agent-Based Framework for Automated Legal Document Analysis and Decision Support.

Two interchangeable pipelines answer the same questions:
  rag      - hybrid retrieval (BM25 + embeddings) with an LLM agent for routing, rewriting and checking
  jev_okf  - knowledge compiled into an Open Knowledge Format (OKF) bundle, navigated with typed
             decisions from Jev (TypeSafe's System One model); the LLM only writes the final answer
"""
