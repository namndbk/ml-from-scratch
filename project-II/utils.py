import numpy as np
import pickle
import re
import pandas as pd
import csv


stop_words = {"should've", 'is', 'all', 'up', 'weren', 'she', 've', "isn't", 'between', 'shan', 'we', "you'd", 'once',
              'where', 'haven', 'herself', 'have', 'hers', 'for', "hasn't", "wouldn't", 'such', 'mustn', 'no', 'was',
              'or', 'my', 'just', "mustn't", 'doesn', 'some', 'them', 'y', 'over', 'didn', 'here', 'does', 'did', 'the',
              "won't", 'their', 'each', 'm', 'those', 'yourselves', 'then', 'who', 'are', 'wasn', 'than', 'won',
              'during', "doesn't", 'not', 'hasn', 'to', 'now', "mightn't", 'these', 'your', 'yours', 'as', 'how',
              "hadn't", 'himself', 'against', 'its', 'that', 'they', 'be', "you're", 'mightn', 'hadn', 'will', "shan't",
              'above', 'ourselves', 'ma', 'while', 'aren', 'after', 'has', "needn't", "wasn't", 'if', 'very', 'off',
              't', "you've", 'out', 'most', "haven't", 'down', 'll', 'further', 'so', 'nor', 'in', 'themselves',
              'needn', 'do', 'myself', 'can', 'ain', 'on', 'own', 'again', "you'll", 'being', 'yourself', "that'll",
              'a', 'were', 'his', 'which', 'of', 'same', 'from', 'me', 'i', 'more', 'but', 'isn', 'our', "it's",
              'other', 'am', 'why', 'shouldn', 's', 'and', 'doing', 'd', 'too', "aren't", 'whom', 'been', 'when',
              'him', 'below', "weren't", "don't", 're', 'about', 'because', 'what', "shouldn't", 'both', 'by', 'into',
              "she's", 'few', "didn't", 'having', 'with', 'under', 'you', 'itself', 'only', 'o', 'an', 'he', 'it',
              'wouldn', 'don', 'theirs', 'her', 'this', 'should', 'there', 'at', 'until', 'any', 'through', "couldn't",
              'before', 'had', 'ours', 'couldn'}

correct_mapping = {
    "Subject": "",
    "subject": ""
}


def remove_special(text):
    sent = re.sub("%|:|'|,|\"|\(|\) |\)|\*|-|(http\S+)|(@\S+)|RT|\#|!|:|\.|[0-9]|\/|\. |\.|\“|’s|;|–|” |\\n|&|-|--", '',
                  text)
    sent = sent.split()
    for i in range(len(sent)):
        if sent[i] in correct_mapping:
            sent[i] = correct_mapping[sent[i]]
    return ' '.join(sent)


def preprocess(sentence):
    sentence = remove_special(sentence)
    sentence_words = []
    sentence = sentence.split()
    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            sentence_words.append(word)
        sentence_words.append(word)
    return sentence_words


def build_dict(sentences):
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
    print('Done !')
    return DICT


def parse(filepath: str, has_header: bool = True):
    sents, lables = list(), list()
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        if has_header:
            next(reader, None)  # skip the headers
        for row in reader:
            sents.append(preprocess(row[0]))
            lables.append(row[1])

    return sents, lables



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