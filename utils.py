import numpy as np
import pickle
from pyvi import ViTokenizer
import re
import pandas as pd

correct_mapping = {
    "Subject": "",
    "subject": ""
}

def remove_special(text):
    sent = re.sub("%|:|'|,|\"|\(|\) |\)|\*|-|(http\S+)|(@\S+)|RT|\#|!|:|\.|[0-9]|\/|\. |\.|\“|’s|;|–|” |\\n|&|-|--",'', text)
    sent = sent.split()
    for i in range(len(sent)):
        if sent[i] in correct_mapping:
            sent[i] = correct_mapping[sent[i]]
    return ' '.join(sent)

def preprocess(sentence):
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
    sentences = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(line[0])
            sentences.append((line[-1]))
    return sentences, labels


def parse(file_name):
    data = pd.read_csv(file_name)
    sents, lables = [], []
    for i in range(data.shape[0]):
        sents.append(preprocess(data.iloc[i, 0]))
        lables.append(data.iloc[i, -1])
    return sents, lables


def build_dict(sentences, max_df):
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
    vector = np.zeros(len(DICT))
    for token in sentence:
        if token in DICT:
            vector[DICT[token]] += 1
        else:
            continue
    return vector

if __name__ == '__main__':
    DICT = load_dict()
    for x in DICT:
        print(x)