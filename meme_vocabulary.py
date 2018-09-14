import nltk
import pickle
import os
import string
import numpy as np
from collections import Counter
from memelookup import MEME
from stop_words import get_stop_words
from nltk.tokenize.casual import TweetTokenizer
import re
tweet_tokenizer = TweetTokenizer()

class Vocabulary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_matrix = []
        self.index = 0



    def add_word(self,word,vector):
        # if the word isn't in dictionary, go ahead and add
        if not word in self.word_to_index.keys():
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.embedding_matrix.append(vector)
            self.index += 1


    def __call__(self, word):
        if not word in self.word_to_index:
            return self.word_to_index['<unk>'], self.embedding_matrix[self.word_to_word['<unk>']]
        return self.word_to_index[word], self.embedding_matrix[self.word_to_index[word]]

    def __len__(self):
        return len(self.word_to_index)

def make_vocab(json,embedding_path):
    # meme = MEME(json)
    # ids = meme.caps.keys()
    # counter = Counter()

    words = []
    max_line_num = 2000
    contract = list(get_stop_words('en'))
    stop = list(string.punctuation) + list(string.digits)
    vocab = Vocabulary()
    meta_words = ['<pad>','<start>','<pause>','<end>','<unk>']
    rand_state = np.random.RandomState(42)

    for index, meta in enumerate(meta_words):
        words.append(meta)
        vector = rand_state.normal(scale=0.6, size=(300, ))
        vocab.add_word(meta,vector)


    for i,w in enumerate(contract):
       w = w.replace("'","")
       if w not in words:
           words.append(w)
           vector = rand_state.normal(scale=0.6, size=(300, ))
           vocab.add_word(w,vector)


    with open(embedding_path) as f:

        for line_num, line in enumerate(f):
            if line_num == max_line_num:
                break

            values = line.split()  # Splits on spaces.
            word = values[0]
            #print(tweet_tokenizer.tokenize(word))
            #print([char for char in word if char not in stop])
            matching = ''.join([char for char in word if char not in stop])
            matching = ''.join(filter(str.isalpha, matching))
            match = (len(matching) == len(word))
            if match:
                vector = np.asarray(values[1:], dtype='float32')
                vocab.add_word(word,vector)



    #print(vocab.word_to_index['one'],vocab.word_to_index['does'],vocab.word_to_index['not'],vocab.word_to_index['simply'])
    print(len(vocab.embedding_matrix))
    print(len(vocab.word_to_index))
    return vocab



def main(json, vocab_path):
    embedding_path = '/Users/mcassettix/github/final_project/project_kojak' + '/data/' + 'glove.42B.300d.txt'
    vocab = make_vocab(json,embedding_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)



if __name__ == '__main__':
    current_dir = os.getcwd()
    json = current_dir + '/captions.json'
    vocab_path = current_dir + '/vocab.pkl'
    main(json, vocab_path)
