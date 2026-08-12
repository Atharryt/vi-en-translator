'EOF'
"""

"""

import numpy as np
import voyageai

GLOSSARY = [
    {"vi": "tri tue nhan tao", "en": "artificial intelligence (AI)"},
    {"vi": "cong nghe thong tin", "en": "information technology (IT)"},
    {"vi": "ca phe sua da", "en": "iced milk coffee (Vietnamese-style)"},
    {"vi": "mang xa hoi", "en": "social media"},
]

def embed_glossary(voyage_client):
    """TRACK 1: Embeddings — run once at startup, not per-request."""
    texts = [f"{g['vi']} = {g['en']}" for g in GLOSSARY]
    result = voyage_client.embed(texts, model="voyage-4", input_type="document")
    return np.array(result.embeddings)

def retrieve_glossary_hits(voyage_client, glossary_vectors, query_text, top_k=2, threshold=0.5):
    """TRACK 1: Vector similarity search — cosine similarity via dot product
    (Voyage embeddings are pre-normalized, so dot product = cosine similarity)."""
    query_vector = voyage_client.embed([query_text], model="voyage-4", input_type="query").embeddings[0]
    similarities = glossary_vectors @ np.array(query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [GLOSSARY[i] for i in top_indices if similarities[i] >= threshold]