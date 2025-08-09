from enum import Enum

class EmbeddingModels(str, Enum):
    tfidf = "TF-IDF"
    word2vec_cbow = "Word2Vec CBOW"
    word2vec_skipgram = "Word2Vec Skip-Gram"