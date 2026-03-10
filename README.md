# AI Talent Intelligence Platform

An AI-powered recruitment assistant that helps recruiters analyze resumes using Retrieval-Augmented Generation (RAG).

This project allows recruiters to upload resumes, rank candidates against a job description, and ask natural language questions about candidates using an AI chatbot.

---

## Features

* Resume parsing from PDF files
* Semantic search using Sentence Transformers
* FAISS vector database for fast similarity search
* AI recruiter chatbot powered by Groq + Llama3
* Streamlit dashboard for interactive use
* Candidate ranking based on job description
* Resume preview inside the dashboard

---

## Demo

Once deployed, the application will be available at:

```
https://yourusername-ai-talent-intelligence.streamlit.app
```

---

## Tech Stack

* Python
* Streamlit
* Sentence Transformers
* FAISS
* Groq API (Llama3)
* NumPy
* PyPDF
* Scikit-learn

---

## Project Structure

```
AI-Talent-Intelligence
│
├── app.py                # Streamlit dashboard
├── rag_chatbot.py        # RAG pipeline + Groq LLM
├── resume_parser.py      # Extract text from PDF resumes
├── embeddings.py         # Generate embeddings
├── ranking.py            # Candidate ranking logic
├── requirements.txt
└── resumes/              # Example resume files
```

---

## How It Works

1. Upload resumes through the Streamlit dashboard.
2. Resume text is extracted from PDFs.
3. Sentence Transformers generate embeddings for each resume.
4. FAISS stores embeddings for fast similarity search.
5. When a recruiter asks a question:

   * The question is embedded.
   * Relevant resumes are retrieved.
   * Groq Llama3 generates an answer using the retrieved context.

This process is called **Retrieval-Augmented Generation (RAG).**

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/AI-Talent-Intelligence.git
```

Navigate to the project folder:

```
cd AI-Talent-Intelligence
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Set API Key

This project uses the Groq API.

Set the environment variable:

Windows (PowerShell)

```
$env:GROQ_API_KEY="your_api_key_here"
```

---

## Run the Application

Start the Streamlit dashboard:

```
streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

---

## Example Questions

You can ask questions like:

* Which candidate has NLP experience?
* Who has Python and Machine Learning skills?
* Which resume mentions Bindu?
* Who is the best match for this job description?

---

## Future Improvements

* Skill extraction from resumes
* Candidate scoring system
* Resume summarization
* Chat history for recruiters
* Integration with ATS systems

---

## Author

Bindu Sri
AI & Data Science Student
Aspiring AI / Data Engineer
