import logging
from datetime import datetime
import pandas as pd
import time

class Logger:
    def __init__(self, log_dir):
        self.name = log_dir + "log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # logging.basicConfig(filename=self.name, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    @staticmethod
    def set_params(params):
        for param in params:
            print(param)
            # logging.info(param)

        # logging.info(100 * "-" + "\n\n")
        print("\n" + 100 * "-" + "\n")

    @staticmethod
    def append(x):
        # logging.info(str(x))
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
        # sentence = str(sentence).translate(punct_translator).translate(digit_translator).replace("\n", "").lower()
        sentence = str(sentence).translate(digit_translator).replace("\n", "")

        sentence = re.sub('\s+', ' ', sentence).strip()

        if len(sentence.split(" ")) > 2:
            t.append({"sentence": sentence, "caseid": caseid})
            
    return t
    
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))