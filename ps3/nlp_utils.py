import logging
from datetime import datetime
import pandas as pd
import time
import nltk
import spacy
import string
import re

digit_translator = str.maketrans('', '', string.digits)
translator = str.maketrans('', '', string.punctuation)
stemmer = nltk.stem.SnowballStemmer('english')
stop = set(nltk.corpus.stopwords.words('english'))


class Constants(object):
    
    DATA_DIR = "./data/"
    RESULT_DIR = "./results/"
    PS = "ps3_"


class Logger:
    def __init__(self, log_dir="./logs/", file=False):
        self.file = file
        
        if self.file:
            self.name = log_dir + "log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logging.basicConfig(filename=self.name, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    def set_params(self, params):
        for param in params:
            print(param)
            
            if self.file:
                logging.info(param)
                
        if self.file:
            logging.info(100 * "-" + "\n\n")
            
        print("\n" + 100 * "-" + "\n")

    def append(self, x):
        if self.file:
            logging.info(str(x))
        print(x)

    @staticmethod
    def create_log():
        logging.shutdown()

        
def save_csv(X, name, h, id_h, id_df, dirpath):
    if id_h != 0:
        id_df = pd.DataFrame({id_h: [int(idx) for idx in range(X.shape[0])]})

    pd.concat([id_df, pd.DataFrame(X, columns=h)], axis=1).to_csv(dirpath + name + '.csv', index=False)
    print(dirpath, name, "created")


def load_csv_to_numpy(X_file, id_h):
    return pd.read_csv(X_file).drop(id_h, axis=1).values


def load_id_df(X_file, id_h):
    return pd.read_csv(X_file)[id_h]


def tokenize_sentence(text, caseid, package="spacy"):
    
    if package == "spacy":
        nlp = spacy.load("en", disable = ["tagger", "ner"])
        doc = nlp(text)
        sentences = doc.sents
        
    elif package == "nltk":
        sentences = nltk.sent_tokenize(text)
        
    t = []
    for sentence in sentences:
        sentence = str(sentence).translate(digit_translator).replace("\n", "")

        sentence = re.sub('\s+', ' ', sentence).strip()

        if len(sentence.split(" ")) > 2:
            t.append({"sentence": sentence, "caseid": caseid})
            
    return t
    
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.tsplit = 0

    def __enter__(self):
        self.tstart = time.time()
        self.tsplit = self.tstart
        
    def __exit__(self):
        print('Elapsed: %s' % (time.time() - self.tstart))
        
    def __split__(self):
        t = time.time()
        print("Elapsed: %s" % (t - self.tsplit))
        self.tsplit = t

        
def get_sentences(doc):
    sentences = []
    
    for raw in nltk.sent_tokenize(doc):
        raw2 = [token for token in raw.translate(translator).lower().split() 
                if token not in stop and len(token) < 10 and len(token) > 2]

        sentences.append(raw2)
    return sentences