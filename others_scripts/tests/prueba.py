from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. ENCODE
emb1 = model.encode("El cielo está despejado")
emb2 = model.encode("No hay nubes en el firmamento")
print(f"Embeddings:\n{emb1}\n{emb2}")
# 2. COSINE SIMILARITY
cos_sim = util.cos_sim([emb1], [emb2])
print(f"Cosine Similarity: {cos_sim}") # Resultado cercano a 1

print(f"Similitud: {cos_sim[0][0]}") # Resultado cercano a 1