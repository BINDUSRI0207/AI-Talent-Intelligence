import streamlit as st
import os
import numpy as np
import faiss

from resume_parser import parse_resumes
from embeddings import get_embedding
from rag_chatbot import build_vector_store, ask_question, generate_answer


st.set_page_config(page_title="AI Talent Intelligence", layout="wide")

st.title(" AI Talent Intelligence Platform")
st.write("Upload resumes, rank candidates, and ask AI recruiter questions.")

# -------------------------------
# Initialize session state
# -------------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

if "names" not in st.session_state:
    st.session_state.names = []

# -------------------------------
# Upload resumes
# -------------------------------
st.header(" Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs("resumes", exist_ok=True)

    for file in uploaded_files:
        path = os.path.join("resumes", file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

    st.success(f"{len(uploaded_files)} resumes uploaded successfully.")


# -------------------------------
# Build Vector Store
# -------------------------------
st.header("Build Resume Index")

if st.button("Build Resume Index"):

    resumes = parse_resumes("resumes")

    index, texts, names = build_vector_store(resumes)

    st.session_state.index = index
    st.session_state.texts = texts
    st.session_state.names = names

    st.success(f"Indexed {len(names)} resumes.")


# -------------------------------
# Candidate Ranking
# -------------------------------
st.header("Candidate Ranking")

job_description = st.text_area(
    "Paste Job Description",
    placeholder="Example: Looking for an AI engineer with Python, NLP and ML experience"
)

if job_description and st.session_state.index:

    job_vector = np.array(get_embedding(job_description), dtype="float32").reshape(1, -1)

    D, I = st.session_state.index.search(job_vector, len(st.session_state.names))

    for rank, idx in enumerate(I[0]):
        st.write(
            f"**Rank {rank+1}: {st.session_state.names[idx]}** "
            f"(distance: {D[0][rank]:.4f})"
        )


# -------------------------------
# Resume Preview
# -------------------------------
st.header(" Resume Preview")

if st.session_state.names:

    selected = st.selectbox(
        "Select resume",
        st.session_state.names
    )

    if selected:

        idx = st.session_state.names.index(selected)

        st.text_area(
            "Resume Content",
            st.session_state.texts[idx],
            height=300
        )


# -------------------------------
# Recruiter AI Chat
# -------------------------------
st.header("Recruiter AI Chat")

question = st.text_input("Ask a recruiter question")

if st.button("Ask"):

    if st.session_state.index is None:
        st.warning("Please build the resume index first.")

    else:

        context = ask_question(
            question,
            st.session_state.index,
            st.session_state.texts,
            st.session_state.names
        )

        answer = generate_answer(context, question)

        st.success(answer)