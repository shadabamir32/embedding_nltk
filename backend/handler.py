from enums.embeddingmodels import EmbeddingModels
from libraries.ntlk_tokenizer import preprocess
from backend.models.tfidf import TF_IDF
import pandas as pd
import json
import os
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

CLEARN_DATA_DIR = "processed_files"

def train_model(file_hash: str, model: EmbeddingModels):
    return get_model(model).load_file(file_hash).fit()

def get_model(model: EmbeddingModels):
    instance = None
    if model == EmbeddingModels.tfidf:
        instance = TF_IDF()
    else:
        instance = TF_IDF()
    return instance

def cosin_lookup(query: str, file_hash: str,model: EmbeddingModels):
    return get_model(model).load_file(file_hash).load_joblib(file_hash).cosin_lookup(preprocess(query))


async def preprocess_json(file_hash: str):
    print('Create Clean data folder')
    os.makedirs(f"{CLEARN_DATA_DIR}/{file_hash}", exist_ok=True)
    print('Reading JSON')
    data_set = pd.read_json(f"uploads/{file_hash}/{file_hash}.json", lines=True)
    print('Preprocessing JSON')
    data_set['short_description'] = data_set['short_description'].parallel_apply(preprocess)
    print('\nSaving as output JSON')
    data_set.to_json(f'{CLEARN_DATA_DIR}/{file_hash}/output.json', orient="records", lines=True)
    print('Preprocessing completed')

# for record in records:
#     record['link'] = record['link']
#     # Preprocessing
#     p_head = preprocess(record['headline'])
#     p_category = preprocess(record['category'])
#     p_desc = preprocess(record['short_description'])
#     # Filtered
#     record['headline'] = p_head['filtered']
#     record['category'] = p_category['filtered']
#     record['short_description'] = p_desc['filtered']
#     # POS
#     # record['pos_headline'] = p_head['tagged']
#     # record['pos_category'] = p_category['tagged']
#     # record['pos_short_description'] = p_desc['tagged']
#     # NER
#     # record['ner_headline'] = p_head['entities']
#     # record['ner_category'] = p_category['entities']
#     # record['ner_short_description'] = p_desc['entities']
#     # bigram
#     # record['bi_headline'] = p_head['bigram']
#     # record['bi_category'] = p_category['bigram']
#     # record['bi_short_description'] = p_desc['bigram']
#     # trigram
#     # record['tri_headline'] = p_head['trigram']
#     # record['tri_category'] = p_category['trigram']
#     # record['tri_short_description'] = p_desc['bigram']
#     # Additional
#     record['authors'] = record['authors']
#     record['date'] = record['date']
# uploadPath = f"{CLEARN_DATA_DIR}/{file_hash}"
# os.makedirs(uploadPath, exist_ok=True)
# with open(f"{uploadPath}/clean_data.json", 'w') as f:
#     for record in records:
#         f.write(json.dumps(record) + '\n')