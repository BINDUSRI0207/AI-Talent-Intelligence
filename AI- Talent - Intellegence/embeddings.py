# embeddings.py
# Converts text into numerical embedding vectors using a pre-trained sentence transformer model.

from sentence_transformers import SentenceTransformer

# Load the pre-trained model once at module level so it's reused across calls.
# 'all-MiniLM-L6-v2' is a lightweight, fast model that produces 384-dimensional embeddings.
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """
    Convert a text string into an embedding vector.

    Args:
        text (str): The input text to embed.

    Returns:
        numpy.ndarray: A 384-dimensional embedding vector representing the text.
    """
    # Encode the text into a dense vector.
    # convert_to_numpy=True ensures we get a numpy array back (default behavior).
    embedding = model.encode(text, convert_to_numpy=True)

    return embedding


# --- Example usage ---
if __name__ == "__main__":
    sample_text = "Machine learning is transforming the world."
    vector = get_embedding(sample_text)

    print(f"Text:            {sample_text}")
    print(f"Embedding shape: {vector.shape}")   # Expected: (384,)
    print(f"First 5 values:  {vector[:5]}")