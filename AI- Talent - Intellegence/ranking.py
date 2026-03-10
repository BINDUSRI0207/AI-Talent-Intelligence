# ranking.py
# Ranks job candidates by comparing their resumes to a job description
# using sentence embeddings and cosine similarity.

from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_embedding


def rank_candidates(resumes: dict, job_description: str) -> list[tuple]:
    """
    Rank resumes by their semantic similarity to a job description.

    Args:
        resumes (dict): A dictionary where each key is a resume filename
                        (e.g. "resume1.pdf") and each value is the resume
                        text content (str).
        job_description (str): The full text of the job description.

    Returns:
        list[tuple]: A list of (resume_filename, similarity_score) tuples,
                     sorted from highest to lowest similarity.
                     Example: [("resume1.pdf", 0.82), ("resume2.pdf", 0.71)]
    """

    # Step 1: Convert the job description into an embedding vector.
    # reshape(1, -1) formats it as a 2D array, which sklearn expects.
    job_embedding = get_embedding(job_description).reshape(1, -1)

    results = []

    # Step 2: Loop through each resume and compute its similarity to the job.
    for resume_name, resume_text in resumes.items():

        # Convert the resume text into an embedding vector.
        resume_embedding = get_embedding(resume_text).reshape(1, -1)

        # Step 3: Compute cosine similarity between the job and resume embeddings.
        # cosine_similarity returns a 2D array — [[score]] — so we extract the float.
        score = cosine_similarity(job_embedding, resume_embedding)[0][0]

        # Step 4: Store the result as a (filename, score) tuple.
        results.append((resume_name, round(float(score), 4)))

    # Step 5: Sort candidates from highest similarity score to lowest.
    ranked = sorted(results, key=lambda x: x[1], reverse=True)

    # Step 6: Return the final ranked list.
    return ranked


# --- Example usage ---
if __name__ == "__main__":
    # Sample job description
    job_description = """
    We are looking for a Python developer with experience in machine learning,
    data preprocessing, and building REST APIs. Knowledge of scikit-learn,
    pandas, and cloud platforms is a plus.
    """

    # Sample resumes (filename → text)
    resumes = {
        "alice.pdf": """
            Experienced Python developer with 4 years in machine learning and data science.
            Proficient in scikit-learn, pandas, and deploying models via REST APIs on AWS.
        """,
        "bob.pdf": """
            Frontend developer skilled in React and TypeScript. Built several e-commerce
            websites and has experience with UI/UX design and CSS frameworks.
        """,
        "carol.pdf": """
            Data analyst with Python scripting experience. Familiar with pandas and
            basic machine learning concepts. Recently completed a cloud computing course.
        """,
    }

    # Rank the candidates
    ranked_candidates = rank_candidates(resumes, job_description)

    # Display results
    print("Candidate Rankings (Best Match First):\n")
    for rank, (name, score) in enumerate(ranked_candidates, start=1):
        print(f"  {rank}. {name:15s} — Similarity Score: {score:.4f}")