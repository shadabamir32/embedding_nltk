import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import HTTPException, BackgroundTasks
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class TF_IDF:
    chunksize = 100_000          # process 100k rows at a time
    max_features = 50000         # vocab size cap
    ngram_range = (1, 2)         # unigrams + bigrams
    sample_size_for_vocab = 1_000_000  # rows for fitting vocabulary
    def __init__(self):
        self.model = TfidfVectorizer(
            max_features= self.max_features,
            ngram_range=self.ngram_range,
            min_df=2
        )

    def load_file(self, file_hash: str, column_name: str):
        if not os.path.exists(f"processed_files/{file_hash}/output.json"):
            raise HTTPException(status_code=404, detail="Your file is still being processed, try again in a minute. If the problem persists, please try to upload file again.")
        self.file_hash = file_hash
        data_set = pd.read_json(f"processed_files/{file_hash}/output.json", lines=True, nrows=self.sample_size_for_vocab)
        if column_name not in data_set.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found in the dataset.")
        self.corpus = data_set[column_name]
        self.column_name = column_name
        print(f"Loaded corpus {self.corpus.__len__()} records")

        return self
    def fit(self):
        # try:
        #     return self.load_joblib(self.file_hash)
        # except Exception:
        #     print('Training model')
        if self.corpus is None or self.corpus.empty:
            print('Corpus not found')
            raise HTTPException(status_code=404, detail="No corpus found.")
        print('Training started')
        self.tfidf_matrix = self.model.fit_transform(self.corpus)
        print('Training finished, dummping results')
        joblib.dump(self.model, f"processed_files/{self.file_hash}/tfidf_vectorizer.pkl")
        joblib.dump(self.tfidf_matrix, f"processed_files/{self.file_hash}/tfidf_matrix.pkl")
        print(f'Result saved at processed_files/{self.file_hash}')

        return self
    def load_joblib(self, file_hash: str):
        if not os.path.exists(f"processed_files/{file_hash}/tfidf_vectorizer.pkl") or not os.path.exists(f"processed_files/{file_hash}/tfidf_matrix.pkl"):
            raise HTTPException(status_code=404, detail="Trained dataset files not found.")
        self.model = joblib.load(f"processed_files/{file_hash}/tfidf_vectorizer.pkl")
        self.tfidf_matrix = joblib.load(f"processed_files/{file_hash}/tfidf_matrix.pkl")

        return self

    def cosin_lookup(self, query: str, top_n: int = 5):
        if self.tfidf_matrix is None:
            print('Trained dataset not loaded.')
            raise HTTPException(status_code=404, detail="Trained dataset not loaded.")
        query_vec = self.model.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top N results
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_scores = similarities[top_indices]

        # Prepare results
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                "doc_index": int(idx),
                "similarity": float(score),
                "word": self.corpus.iloc[idx] if self.corpus is not None else "NA"
            })

        return results