import numpy as np
import pickle
import re
import pandas as pd


from sklearn.datasets import load_iris

correct_mapping = {
    "Subject": "",
    "subject": ""
}

def remove_special(text):
    """
    Remove special character
    @param: text
    @type: str
    @return: text
    @rtype: str
    """
    sent = re.sub("%|:|'|,|\"|\(|\) |\)|\*|-|(http\S+)|(@\S+)|RT|\#|!|:|\.|[0-9]|\/|\. |\.|\“|’s|;|–|” |\\n|&|-|--",'', text)
    sent = sent.split()
    for i in range(len(sent)):
        if sent[i] in correct_mapping:
            sent[i] = correct_mapping[sent[i]]
    return ' '.join(sent)

def preprocess(sentence):
    """
    Preprocessing for text: remove special character, remove stop word, lower
    @param: text, type str
    @return: List of word
    @rtype: List[str] 
    """
    sentence = remove_special(sentence)
    # stop_words = load_stop_words()
    new_sentences = []
    # sentence = ViTokenizer.tokenize(sentence)
    sentence = sentence.split()
    for word in sentence:
        word = word.lower()
        # if word not in stop_words:
        #     new_sentences.append(word)
        new_sentences.append(word)
    return new_sentences


def load_stop_words():
    f = open('stopwords.txt', 'r')
    stop_word = []
    for line in f:
        stop_word.append(line.strip())
    return stop_word


def parse_file(file_name):
    """
    Load data
    """
    sentences = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(line[0])
            sentences.append((line[-1]))
    return sentences, labels


def parse(file_name):
    """
    Load data csv
    @param: path file
    @type: str
    @return:
        sents: List of sentence
        labels: List of label
    @rtype:
        sents: List[str]
        labels: List[str]
    """
    data = pd.read_csv(file_name)
    sents, labels = [], []
    for i in range(data.shape[0]):
        sents.append(preprocess(data.iloc[i, 0]))
        lables.append(data.iloc[i, -1])
    return sents, labels


def build_dict(sentences):
    """"
    Building dictionary for dataset
    @param: List of sentence
    @type: List[str]
    """
    DICT = {}
    count = 0
    print('Building dictionary !')
    for sent in sentences:
        for word in sent:
            if word not in DICT:
                DICT[word] = count
                count += 1
            else:
                continue
    pickle.dump(DICT, open('DICT', 'wb'))
    print('Done !')

def load_dict():
    try:
        DICT = pickle.load(open('DICT', 'rb'))
        return DICT
    except:
        return {}

def bag_of_word(sentence, DICT):
    """
    Create bag_of_word vector. Sentence represented by one vector
    E.g: [0, 1, 3, 4, 0]
    @param:
        sentences: sentence
        DICT: dictionary
    @type:
        sentences: string
        DICT: dict
    @return: vector, element i in vector = count(word i in sentence) (i must in DICT)
    """
    vector = np.zeros(len(DICT))
    for token in sentence:
        if token in DICT:
            vector[DICT[token]] += 1
        else:
            continue
    return vector


def load_iris_data():
    """
    Load iris data with two label
    @return:
        X: data point
        y: label
    @rtype:
        X: narray
        y: array
    """
    data = load_iris()
    _X = data.data
    _y = data.target
    # Get data with label 0, 1
    X = _X[_y < 2]
    y = _y[_y < 2]
    for i, c in enumerate(y):
        if c == 0:
            # transform label 0 to -1
            y[i] = -1
    return X, y


if __name__ == '__main__':
    DICT = load_dict()
    for x in DICT:
        print(x)