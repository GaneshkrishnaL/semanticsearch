import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

def main():
    st.title("Semantic Search with Streamlit")

    # Define your sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language.",
        "Semantic search enhances the relevance of search results.",
        "Artificial intelligence is transforming various industries.",
        "Machine learning algorithms can make predictions from data.",
        "Data science involves extracting insights from data.",
        "The Earth revolves around the Sun.",
        "Neural networks are a key component of deep learning.",
        "Quantum computing has the potential to revolutionize computation.",
        "Climate change is a pressing global issue."
    ]

    # User input for the query
    query = st.text_input("Enter your query:")

    if query:
        # Load a pre-trained Sentence Transformers model
        model = SentenceTransformer("sentence-transformers/msmarco-bert-base-dot-v5")

        # EMBEDDINGS
        query_embedding = model.encode(query, convert_to_tensor=True)
        sentence_embeddings = [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]

        query_embedding = query_embedding.detach().numpy()
        sentence_embeddings = [embedding.detach().numpy() for embedding in sentence_embeddings]

        # cosine similarity
        similarities = [util.pytorch_cos_sim(query_embedding, torch.tensor(sentence_embedding)).item() for sentence_embedding in sentence_embeddings]

        # Get top 3 most similar sentences and their similarity scores
        top_indices = np.argsort(similarities)[-3:][::-1]

        # Display the results
        st.subheader("Top 3 Most Similar Sentences:")
        for idx in top_indices:
            st.write(f"Sentence: {sentences[idx]}")
            st.write(f"Cosine Similarity: {similarities[idx]}")
            st.write("---")

if __name__ == "__main__":
    main()
