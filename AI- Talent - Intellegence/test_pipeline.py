import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resume_parser import parse_resumes
from ranking import rank_candidates


job_description = """
Looking for AI engineer with Python, NLP,
Machine Learning, and Transformers experience.
"""

# Parse resumes
resumes = parse_resumes("resumes")

# Rank candidates
ranking = rank_candidates(resumes, job_description)

print("\nCandidate Ranking:\n")

for name, score in ranking:
    print(name, "Score:", score)
