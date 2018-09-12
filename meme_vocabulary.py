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
        self.index = 0


    def add_word(self,word):
        # if the word isn't in dictionary, go ahead and add
        if not word in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def __call__(self, word):
        if not word in self.word_to_index:
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def __len__(self):
        return len(self.word_to_index)

def make_vocab(json,embedding_path):
    # meme = MEME(json)
    # ids = meme.caps.keys()
    # counter = Counter()

    words = []
    vectors = []
    max_line_num = 1500
    contract = list(get_stop_words('en'))
    stop = list(string.punctuation) + list(string.digits)
    print(stop)
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<pause>') #to deliniate top and bottom meme caption
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

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
                print(word)
                vocab.add_word(word)
                # vector = np.asarray(values[1:], dtype='float32')
                # vectors.append(vector)

        for w in contract:
            w = w.replace("'","")
            if w not in words:
                words.append(w)
                vocab.add_word(w)
                # rand_state = np.random.RandomState(42)
                # vectors.append(rand_state.normal(scale=0.6, size=(300, )))

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
