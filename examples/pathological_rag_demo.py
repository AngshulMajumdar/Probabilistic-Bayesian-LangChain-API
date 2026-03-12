"""Pathological RAG benchmark: greedy single-path vs Bayesian multi-hypothesis."""
from __future__ import annotations

import json
import random
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


stale_docs = [
    {"id": "ST-001", "title": "Old HR Portal Migration Note", "text": "The HR portal is WorkSphere. Employees should use WorkSphere for leave requests and payslips.", "verified": False, "source_type": "stale_cache"},
    {"id": "ST-002", "title": "Legacy Payroll FAQ", "text": "Payroll records may be accessed from WorkSphere after login.", "verified": False, "source_type": "stale_cache"},
]
verified_docs = [
    {"id": "VF-001", "title": "Current Portal Access Policy", "text": "The current HR portal is PeopleHub. Employees must use PeopleHub for leave requests, payslips, and profile updates.", "verified": True, "source_type": "official_policy"},
    {"id": "VF-002", "title": "Official HR Handbook", "text": "All employee self-service workflows are managed through PeopleHub.", "verified": True, "source_type": "official_policy"},
]
query = "What is the name of the HR portal?"
gold_answer = "PeopleHub"


def build_index(documents, retriever):
    texts = [d["title"] + ". " + d["text"] for d in documents]
    embs = retriever.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve_top(documents, index, retriever, query, top_k=1):
    q_emb = retriever.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)
    row = []
    for idx, score in zip(indices[0], scores[0]):
        row.append({**documents[idx], "score": float(score)})
    return row


def generate_answer(context, query, tokenizer, generator, device):
    prompt = f"""Answer the question using only the context below.
Return only the answer phrase.
If the answer is not present, say: unknown.

Context:
{context}

Question: {query}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    with torch.no_grad():
        outputs = generator.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def greedy_rag(query, rng, retriever, stale_index, verified_index, tokenizer, generator, device):
    if rng.random() < 0.7:
        chosen_pool, chosen_index, chosen_name = stale_docs, stale_index, "stale_cache"
    else:
        chosen_pool, chosen_index, chosen_name = verified_docs, verified_index, "official_policy"
    retrieved = retrieve_top(chosen_pool, chosen_index, retriever, query, top_k=1)[0]
    answer = generate_answer(retrieved["text"], query, tokenizer, generator, device)
    success = gold_answer.lower() in answer.lower() or answer.lower() == gold_answer.lower()
    return {"method": "greedy_rag", "chosen_source": chosen_name, "retrieved_doc": retrieved["id"], "retrieved_verified": retrieved["verified"], "answer": answer, "success": success, "tool_calls": 1}


def bayesian_rag(query, retriever, stale_index, verified_index, tokenizer, generator, device):
    stale_hit = retrieve_top(stale_docs, stale_index, retriever, query, top_k=1)[0]
    verified_hit = retrieve_top(verified_docs, verified_index, retriever, query, top_k=1)[0]
    stale_score = stale_hit["score"] + (0.0 if stale_hit["verified"] else -0.8)
    verified_score = verified_hit["score"] + (0.8 if verified_hit["verified"] else 0.0)
    chosen = verified_hit if verified_score >= stale_score else stale_hit
    posterior_verified = 1 / (1 + np.exp(-(verified_score - stale_score) * 5))
    answer = generate_answer(chosen["text"], query, tokenizer, generator, device)
    success = gold_answer.lower() in answer.lower() or answer.lower() == gold_answer.lower()
    return {"method": "bayesian_rag", "chosen_source": chosen["source_type"], "retrieved_doc": chosen["id"], "retrieved_verified": chosen["verified"], "answer": answer, "success": success, "tool_calls": 2, "posterior_verified_source": float(posterior_verified), "stale_score": float(stale_score), "verified_score": float(verified_score)}


def main():
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = generator.to(device)

    stale_index = build_index(stale_docs, retriever)
    verified_index = build_index(verified_docs, retriever)

    rng = random.Random(123)
    single_g = greedy_rag(query, rng, retriever, stale_index, verified_index, tokenizer, generator, device)
    single_b = bayesian_rag(query, retriever, stale_index, verified_index, tokenizer, generator, device)
    print(json.dumps({"single_demo": {"greedy": single_g, "bayesian": single_b}}, indent=2))

    records = []
    for i in range(100):
        rng = random.Random(1000 + i)
        g = greedy_rag(query, rng, retriever, stale_index, verified_index, tokenizer, generator, device)
        b = bayesian_rag(query, retriever, stale_index, verified_index, tokenizer, generator, device)
        records.append({
            "trial": i + 1,
            "greedy_success": g["success"],
            "bayesian_success": b["success"],
            "greedy_source": g["chosen_source"],
            "bayesian_source": b["chosen_source"],
            "greedy_verified_doc": g["retrieved_verified"],
            "bayesian_verified_doc": b["retrieved_verified"],
            "greedy_answer": g["answer"],
            "bayesian_answer": b["answer"],
            "greedy_tool_calls": g["tool_calls"],
            "bayesian_tool_calls": b["tool_calls"],
            "posterior_verified_source": b["posterior_verified_source"],
        })
    df = pd.DataFrame(records)
    summary = {
        "trials": len(df),
        "greedy_success_rate_percent": 100.0 * df["greedy_success"].mean(),
        "bayesian_success_rate_percent": 100.0 * df["bayesian_success"].mean(),
        "greedy_avg_tool_calls": float(df["greedy_tool_calls"].mean()),
        "bayesian_avg_tool_calls": float(df["bayesian_tool_calls"].mean()),
        "greedy_verified_retrieval_rate_percent": 100.0 * df["greedy_verified_doc"].mean(),
        "bayesian_verified_retrieval_rate_percent": 100.0 * df["bayesian_verified_doc"].mean(),
        "avg_posterior_verified_source": float(df["posterior_verified_source"].mean()),
    }
    print(json.dumps(summary, indent=2))
    df.to_csv("pathological_rag_trial_results.csv", index=False)
    with open("pathological_rag_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
