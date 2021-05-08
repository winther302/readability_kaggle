import pandas as pd
import numpy as np
from nltk import word_tokenize

#import csv file return as numpy array
def import_data(file_name:str):
    df = pd.read_csv(file_name)
    return df[['id','excerpt','target']].to_numpy()

def find_vocab(text):
    vocab = set([])
    for excertp in text:
        for sentence in excertp:
            for word in word_tokenize(sentence):
                if word not in vocab:
                    vocab.add(word.lower())
    return vocab

def print_vocab(vocab):
    for word in vocab:
        print(word)

def main():
    data = import_data("data/train.csv")
    vocab = find_vocab(data[:,1:2])
    print_vocab(vocab)


if __name__ == '__main__':
    main()
