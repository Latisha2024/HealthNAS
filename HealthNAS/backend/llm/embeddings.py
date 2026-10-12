from sentence_transformers import SentenceTransformer


class HealthcareEmbeddings:
    """Generate embeddings for healthcare text using MiniLM."""

    def __init__(self):
        print("\nLoading healthcare embedding model...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_embedding_dimension()

        print("Embedding model loaded successfully.")
        print(f"Embedding dimension: {self.dimension}")

    def encode(self, text):
        """Generate a normalized embedding for a text."""
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_single(self, text):
        """Generate an embedding for a single query."""
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vector[0]

    def encode_documents(self, documents):
        """Generate embeddings for multiple documents."""
        return self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def demonstrate_embedding(self, text):
        """Display an example text-to-vector conversion."""
        vector = self.encode(text)

        print("\n" + "=" * 70)
        print("TEXT → NUMERICAL VECTOR")
        print("=" * 70)
        print(f"\nInput text:\n{text}")
        print(f"\nVector dimension: {len(vector)}")
        print("\nFirst 10 vector values:")
        print(vector[:10])

        return vector

    def get_dimension(self):
        return self.dimension

    def close(self):
        """Release the embedding model."""
        try:
            del self.model
            print("Healthcare embedding model released.")
        except Exception:
            pass


HealthcareEmbedding = HealthcareEmbeddings