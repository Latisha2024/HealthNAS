import numpy as np

from .embeddings import HealthcareEmbeddings
from .knowledge_base import HEALTHCARE_KNOWLEDGE


class HealthcareRAG:
    """
    RAG component for retrieving healthcare knowledge
    relevant to the NAS problem.
    """

    def __init__(self):
        # Load the sentence embedding model
        self.embedding_model = HealthcareEmbeddings()

        # Load healthcare documents
        self.documents = HEALTHCARE_KNOWLEDGE

        # Convert all documents into embeddings
        self.document_embeddings = (
            self.embedding_model.encode_documents(
                self.documents
            )
        )

        print("\nHealthcare RAG initialized successfully.")

    def cosine_similarity(self, query_vector, document_vector):
        """Calculate similarity between two embeddings."""

        query_vector = np.asarray(query_vector)
        document_vector = np.asarray(document_vector)

        # Calculate the cosine similarity denominator
        denominator = (
            np.linalg.norm(query_vector)
            * np.linalg.norm(document_vector)
        )

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        # Calculate cosine similarity
        return float(
            np.dot(query_vector, document_vector)
            / denominator
        )

    def retrieve(self, query, top_k=3):
        """
        Retrieve the most relevant healthcare documents
        for the user's machine-learning problem.
        """

        # Convert the user query into an embedding
        query_vector = (
            self.embedding_model.encode_single(query)
        )

        results = []

        # Compare the query with every document
        for index, document in enumerate(self.documents):

            document_vector = self.document_embeddings[index]

            score = self.cosine_similarity(
                query_vector,
                document_vector
            )

            results.append(
                {
                    "document": document,
                    "score": score
                }
            )

        # Rank documents by similarity score
        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Return the most relevant documents
        return results[:top_k]

    def display_results(self, results):
        """Display the documents retrieved by RAG."""

        print("\n")
        print("=" * 70)
        print("RAG RETRIEVED KNOWLEDGE")
        print("=" * 70)

        for i, result in enumerate(results, start=1):

            print(f"\nDocument {i}")

            print(
                f"Similarity Score: "
                f"{result['score']:.4f}"
            )

            print(result["document"])

    def close(self):
        """Release RAG resources."""

        try:
            # Release the embedding model
            if hasattr(self, "embedding_model"):
                self.embedding_model.close()

            # Remove stored document embeddings
            self.document_embeddings = None

            print("Healthcare RAG resources released.")

        except Exception as error:
            print(f"RAG cleanup warning: {error}")