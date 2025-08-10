import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize, TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from typing import List

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# def get_wordnet_pos(word) -> str:
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
#     return tag_dict.get(tag, wordnet.NOUN)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text) -> str:
    tokens = word_tokenize(text.lower())
    #filtered = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    pos_tags = nltk.pos_tag(tokens)
    filtered = [
        lemmatizer.lemmatize(w, get_wordnet_pos(tag))
        #for w in tokens
        for w, tag in pos_tags
        if w.isalpha() and w not in stop_words
    ]
    return " ".join(filtered)
# tagged = nltk.pos_tag(filtered)
# entities = nltk.chunk.ne_chunk(tagged, True)
# bigram = list(nltk.ngrams(filtered, 2))
# trigram = list(nltk.ngrams(filtered, 3))
# print(entities)