"""Colab-friendly normal RAG demo with small open-source models."""
from __future__ import annotations

import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


docs = [
    {"id": "HR-001", "title": "Leave Policy", "text": "Employees are entitled to 18 paid leave days per year. Casual leave and sick leave are tracked in the HR portal."},
    {"id": "HR-002", "title": "POSH Policy", "text": "The organisation follows a zero tolerance policy towards sexual harassment. Complaints may be filed with the Internal Committee through the official ethics portal."},
    {"id": "HR-003", "title": "Portal Access", "text": "The HR portal is PeopleHub. Employees should use PeopleHub for leave requests, payslips, and profile updates."},
    {"id": "HR-004", "title": "Remote Work", "text": "Employees may work remotely up to two days per week with manager approval."},
    {"id": "HR-005", "title": "Payroll Policy", "text": "Salary slips are published on the PeopleHub portal by the fifth working day of each month."},
]


def retrieve(index, retriever, query, top_k=3):
    q_emb = retriever.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append({
            "rank": rank,
            "id": docs[idx]["id"],
            "title": docs[idx]["title"],
            "score": float(score),
            "text": docs[idx]["text"],
        })
    return results


def answer_with_rag(query, index, retriever, tokenizer, generator, device, top_k=3, max_new_tokens=64):
    retrieved = retrieve(index, retriever, query, top_k=top_k)
    context = "\n".join([f"[{r['id']}] {r['title']}: {r['text']}" for r in retrieved])
    prompt = f"""Answer the question using only the context below.
If the answer is not present in the context, say: 'I do not know based on the provided documents.'

Context:
{context}

Question: {query}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = generator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"query": query, "retrieved_chunks": retrieved, "answer": answer}


def main():
    retriever_name = "sentence-transformers/all-MiniLM-L6-v2"
    generator_name = "google/flan-t5-small"
    retriever = SentenceTransformer(retriever_name)
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = generator.to(device)

    doc_texts = [d["title"] + ". " + d["text"] for d in docs]
    doc_embeddings = retriever.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    queries = [
        "What is the name of the HR portal?",
        "How many paid leave days do employees get per year?",
        "Where should employees file POSH complaints?",
        "How many remote work days are allowed each week?",
    ]
    results = [answer_with_rag(q, index, retriever, tokenizer, generator, device) for q in queries]
    print(json.dumps(results, indent=2))
    with open("normal_rag_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
