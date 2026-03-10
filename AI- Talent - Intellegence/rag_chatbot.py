"""
rag_chatbot.py - RAG Chatbot for Resume Analysis

This chatbot helps recruiters query a set of PDF resumes using:
- sentence-transformers for semantic embeddings
- FAISS for fast similarity search
- Groq API (Llama3) for generating natural language answers

Usage:
    python rag_chatbot.py resumes/
"""

import os
import sys
import numpy as np
import faiss

from groq import Groq

# Local modules (assumed to exist in the same directory)
from resume_parser import parse_resumes
from embeddings import get_embedding


# ─────────────────────────────────────────────
# Groq client setup — reads API key from env
# ─────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# STEP 1 & 2 & 3: Build FAISS vector store
# ─────────────────────────────────────────────
def build_vector_store(resumes: dict) -> tuple:
    """
    Convert resume texts into embeddings and store them in a FAISS index.

    Args:
        resumes (dict): { "filename.pdf": "resume text ...", ... }

    Returns:
        index  - FAISS index containing all resume embeddings
        texts  - list of resume text strings (same order as index)
        names  - list of resume filenames (same order as index)
    """
    texts = []   # raw resume texts, kept for context retrieval
    names = []   # filenames, used to tell the recruiter which resume matched
    vectors = [] # embedding vectors that go into FAISS

    print(f"\n📄 Indexing {len(resumes)} resume(s)...")

    for filename, text in resumes.items():
        # Skip empty documents so they don't pollute the index
        if not text.strip():
            print(f"  ⚠️  Skipping '{filename}' — no text found.")
            continue

        # Convert resume text → dense vector using sentence-transformers
        embedding = get_embedding(text)          # returns list or np.array
        embedding = np.array(embedding, dtype="float32")

        vectors.append(embedding)
        texts.append(text)
        names.append(filename)
        print(f"  ✅ Indexed: {filename}")

    if not vectors:
        raise ValueError("No valid resumes were found to index.")

    # Stack all vectors into a 2-D numpy matrix  (num_resumes × embedding_dim)
    matrix = np.vstack(vectors)

    # Create a flat (brute-force) FAISS index using L2 (Euclidean) distance.
    # For a small resume set this is perfectly fast; swap for IndexIVFFlat
    # if you ever need to scale to tens of thousands of documents.
    dimension = matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add all resume vectors to the index in one shot
    index.add(matrix)

    print(f"\n🗂️  FAISS index built — {index.ntotal} vector(s) stored.\n")
    return index, texts, names


# ─────────────────────────────────────────────
# STEP 5–8: Retrieve relevant resumes
# ─────────────────────────────────────────────
def ask_question(question: str, index, texts: list, names: list, top_k: int = 3) -> str:
    """
    Embed the recruiter's question and retrieve the top-k most similar resumes.

    Args:
        question : recruiter's natural-language question
        index    : FAISS index built by build_vector_store()
        texts    : list of resume texts (parallel to index)
        names    : list of filenames  (parallel to index)
        top_k    : how many resumes to retrieve (default 3)

    Returns:
        context  : formatted string combining the top-k resume excerpts
    """
    # Embed the question using the same model used for resumes
    query_vector = np.array(get_embedding(question), dtype="float32").reshape(1, -1)

    # FAISS search — returns distances and integer indices of nearest neighbours
    k = min(top_k, index.ntotal)   # can't retrieve more than what's indexed
    distances, indices = index.search(query_vector, k)

    # Build a readable context block from the retrieved resumes
    context_parts = []
    print(f"\n🔍 Top {k} resume(s) retrieved for your question:")

    for rank, idx in enumerate(indices[0], start=1):
        name = names[idx]
        text = texts[idx]
        print(f"  {rank}. {name}  (L2 distance: {distances[0][rank - 1]:.4f})")

        # Truncate very long resumes to keep the prompt within token limits
        excerpt = text[:3000] + ("..." if len(text) > 3000 else "")
        context_parts.append(f"--- Resume: {name} ---\n{excerpt}")

    context = "\n\n".join(context_parts)
    return context


# ─────────────────────────────────────────────
# STEP 9–10: Generate answer with Groq / Llama3
# ─────────────────────────────────────────────
def generate_answer(context: str, question: str) -> str:
    """
    Send the retrieved resume context + recruiter question to Groq's Llama3
    and return the generated answer.

    Args:
        context  : formatted text from the top retrieved resumes
        question : the recruiter's original question

    Returns:
        answer   : Llama3's natural-language response
    """
    # A clear system prompt helps the model stay in the recruiter-assistant role
    system_prompt = (
        "You are an expert recruiter assistant. "
        "You are given excerpts from candidate resumes and a question from a recruiter. "
        "Answer the question clearly and concisely, citing specific candidates by name where relevant. "
        "If the information is not present in the resumes, say so honestly."
    )

    # Combine context and question into the user message
    user_message = (
        f"Here are the relevant resume excerpts:\n\n"
        f"{context}\n\n"
        f"Recruiter question: {question}"
    )

    # Call Groq's chat completion endpoint with Llama3-70B
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,      # lower = more factual, less creative
        max_tokens=1024,
    )

    # Extract the assistant's text from the response object
    answer = response.choices[0].message.content
    return answer


# ─────────────────────────────────────────────
# MAIN — orchestrates everything
# ─────────────────────────────────────────────
def main():
    """
    Entry point:
      1. Read the resumes folder path from the CLI argument.
      2. Parse resumes → build FAISS index.
      3. Enter an interactive Q&A loop for the recruiter.
    """
    # ── CLI argument: folder that contains the PDF resumes ──
    if len(sys.argv) < 2:
        print("Usage: python rag_chatbot.py <path_to_resumes_folder>")
        print("Example: python rag_chatbot.py resumes/")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"❌ Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    # ── Check that the Groq API key is set ──
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is not set.")
        print("   Export it with: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)

    # ── Step 1: Load resumes from the folder ──
    print(f"\n📂 Loading resumes from: {folder_path}")
    resumes = parse_resumes(folder_path)

    if not resumes:
        print("❌ No resumes found in the specified folder.")
        sys.exit(1)

    print(f"   Found {len(resumes)} resume file(s).")

    # ── Steps 2–4: Build FAISS vector store ──
    index, texts, names = build_vector_store(resumes)

    # ── Steps 5–10: Interactive recruiter Q&A loop ──
    print("=" * 60)
    print("🤖 Resume RAG Chatbot — powered by Llama3 via Groq")
    print("   Type your question and press Enter.")
    print("   Type 'exit' to quit.")
    print("=" * 60)

    while True:
        # Prompt the recruiter for a question
        question = input("\nAsk recruiter AI: ").strip()

        # Allow the user to exit cleanly
        if question.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        # Skip blank input
        if not question:
            print("⚠️  Please enter a question.")
            continue

        try:
            # Retrieve relevant resume context via FAISS
            context = ask_question(question, index, texts, names)

            # Generate and print the answer
            print("\n💬 Answer:\n")
            answer = generate_answer(context, question)
            print(answer)
            print("\n" + "-" * 60)

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("   Please try again or type 'exit' to quit.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()