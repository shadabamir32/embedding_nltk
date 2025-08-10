from enums.embeddingmodels import EmbeddingModels
from libraries.ntlk_tokenizer import preprocess
from backend.models.tfidf import TF_IDF
from backend.models.word2vec import Word2VecModel
import pandas as pd
import json
import os
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

CLEARN_DATA_DIR = "processed_files"


def train_model(file_hash: str, column_name: str, model: EmbeddingModels):
    if model == EmbeddingModels.tfidf:
        instance = TF_IDF()
        print('Loading file for TF-IDF model')
        instance.load_file(file_hash, column_name).fit()
    elif model == EmbeddingModels.word2vec_cbow:
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=0)
        print('Loading file for Word2Vec CBOW model')
        instance.load_file(file_hash, column_name).fit()
    elif model == EmbeddingModels.word2vec_skipgram:
        print('Loading file for Word2Vec Skip-gram model')
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=1)
        instance.load_file(file_hash, column_name).fit()
    else:
        raise ValueError(f"Unsupported model type: {model}")


def cosin_lookup(query: str, file_hash: str, column_name: str, model: EmbeddingModels):
    result = None
    if model == EmbeddingModels.tfidf:
        instance = TF_IDF()
        result = instance.load_file(file_hash, column_name).load_joblib(
            file_hash).cosin_lookup(preprocess(query))
    elif model == EmbeddingModels.word2vec_cbow:
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=0)
        result = instance.load_file(file_hash, column_name).load_joblib(
            file_hash).most_similar(preprocess(query))
    elif model == EmbeddingModels.word2vec_skipgram:
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=1)
        result = instance.load_file(file_hash, column_name).load_joblib(
            file_hash).most_similar(preprocess(query))
    else:
        raise ValueError(f"Unsupported model type: {model}")
def reduce_embeddings(file_hash: str, column_name: str, model: EmbeddingModels):
    result = None
    if model == EmbeddingModels.tfidf:
        instance = TF_IDF()
        print('Loading file for TF-IDF model')
        result = instance.load_file(file_hash, column_name).load_joblib(file_hash).get_embeddings()
    elif model == EmbeddingModels.word2vec_cbow:
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=0)
        result = instance.load_file(file_hash, column_name).load_joblib(file_hash).get_embeddings()
    elif model == EmbeddingModels.word2vec_skipgram:
        instance = Word2VecModel(vector_size=1000, window=5, min_count=1, sg=1)
        result = instance.load_file(file_hash, column_name).load_joblib(file_hash).get_embeddings()
    else:
        raise ValueError(f"Unsupported model type: {model}")
    return result


async def preprocess_json(file_hash: str, column_name: str):
    print('Create Clean data folder')
    os.makedirs(f"{CLEARN_DATA_DIR}/{file_hash}", exist_ok=True)
    print('Reading JSON')
    data_set = pd.read_json(
        f"uploads/{file_hash}/{file_hash}.json", lines=True)
    print('Preprocessing JSON')
    data_set[column_name] = data_set[column_name].parallel_apply(
        preprocess)
    print('\nSaving as output JSON')
    data_set.to_json(
        f'{CLEARN_DATA_DIR}/{file_hash}/output.json', orient="records", lines=True)
    print('Preprocessing completed')