import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize, TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# english_stops = set(stopwords.words('english'))
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text) -> str:
    tokens = word_tokenize(text.lower())
    filtered = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)
# tagged = nltk.pos_tag(filtered)
# entities = nltk.chunk.ne_chunk(tagged, True)
# bigram = list(nltk.ngrams(filtered, 2))
# trigram = list(nltk.ngrams(filtered, 3))
# print(entities)