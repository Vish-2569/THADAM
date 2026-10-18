from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading multilingual model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

complaints = [
    "There is waterlogging on the road.",
    "Road la water stagnate aagudhu.",
    "Garbage has accumulated near the bus stop."
]

embeddings = model.encode(complaints)

similarities = cosine_similarity(embeddings)

print("\nEmbedding shape:", embeddings.shape)
print("\nSimilarity matrix:")
print(similarities)

print("\nModel test completed.")