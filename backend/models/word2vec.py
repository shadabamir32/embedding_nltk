from fastapi import HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

class Word2VecModel:
    sample_size_for_vocab = 1_000_000  # rows for fitting vocabulary
    def __init__(self, vector_size: int = 1000, window: int = 5, min_count: int = 1, sg: int = 0):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None
        self.corpus = None
        self.file_hash = None

    def fit(self):
        if self.corpus is None:
            raise HTTPException(status_code=404, detail="No corpus found.")
        self.model = Word2Vec(
            sentences=self.corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg
        )
        print("Training finished, saving model...")
        self.model.save(f"processed_files/{self.file_hash}/word2vec_sg_{self.sg}.model")
        print(f"Model saved at processed_files/{self.file_hash}")
        return self

    def load_file(self, file_hash: str, column_name: str):
        path = f"processed_files/{file_hash}/output.json"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Processed file not found.")

        self.file_hash = file_hash
        print(f"Loading file {path}")
        # Load the JSON file and preprocess the text
        df = pd.read_json(path, lines=True, nrows=self.sample_size_for_vocab)
        # Preprocess the text data
        print("Preprocessing text data...")
        if column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found in the dataset.")
        self.corpus = [simple_preprocess(str(text)) for text in df[column_name]]
        print(f"Loaded corpus with {len(self.corpus)} records")

        return self

    def load_joblib(self, file_hash: str):
        path = f"processed_files/{file_hash}/word2vec_sg_{self.sg}.model"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Trained dataset files not found.")
        
        self.model = Word2Vec.load(path)
        self.file_hash = file_hash
        return self

    # def get_vector(self, word):
    #     return self.model.wv[word] if word in self.model.wv else None

    def most_similar(self, query: str, top_n: int = 5):
        if not self.model:
            raise HTTPException(status_code=404, detail="Trained dataset not loaded.")
        similar_words = self.model.wv.most_similar(query, topn=top_n) if query in self.model.wv else []
        results = []
        for word, score in similar_words:
            results.append({
                "word": word,
                "similarity": float(score)
            })

        return results
    def get_embeddings(self):
        print("Generating embeddings for all words in the vocabulary...")
    
    def cosin_lookup(self, query: str, top_n: int = 5):
        if not self.model:
            raise HTTPException(status_code=404, detail="Trained dataset not loaded.")

        tokens = simple_preprocess(query)
        word_vectors = []
        for token in tokens:
            if token in self.model.wv:
                word_vectors.append(self.model.wv[token])

        if not word_vectors:
            raise HTTPException(status_code=404, detail="No words from query in vocabulary.")

        query_vec = np.mean(word_vectors, axis=0).reshape(1, -1)

        # Compute similarity with all words in vocab
        vocab_words = list(self.model.wv.index_to_key)
        vocab_vectors = np.array([self.model.wv[word] for word in vocab_words])
        similarities = cosine_similarity(query_vec, vocab_vectors).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_n]
        results = []
        for idx in top_indices:
            results.append({
                "word": vocab_words[idx],
                "similarity": float(similarities[idx])
            })

        return results