import nltk
import pickle
import os
import string
import numpy as np
from collections import Counter
from memelookup import MEME
from stop_words import get_stop_words
from textblob import TextBlob
from nltk.tokenize.casual import TweetTokenizer

import re
tweet_tokenizer = TweetTokenizer()

class Vocabulary(object):
    def __init__(self, json, embedding_path):
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_matrix = []
        self.index = 0
        self.make_vocab(json, embedding_path)


    def __call__(self, word):
        if not word in self.word_to_index:
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def __len__(self):
        return len(self.word_to_index)

    def make_vocab(self, json, embedding_path):
        meme = MEME(json)
        ids = meme.caps.keys()
        #stop = list(string.punctuation) #+ list(string.digits)
        counter = Counter()
        max_meme_vocab = 6000
        for i, id in enumerate(ids):
            caption = str(meme.caps[id]['caption'])
            caption = caption.replace("'","")
            caption = caption.split(' <pause> ')
            #print(caption)
            upper_caption = caption[0]
            lower_caption = caption[-1]

            # text = text.replace('—', ' — ')
            # text = text.replace('-', ' - ')

            #print(list(tuple(TextBlob(upper_caption).tokens)))
            #print(list(tuple(TextBlob(lower_caption).tokens)))
            upper_cap_list = list(tuple(TextBlob(upper_caption).tokens))
            lower_cap_list = list(tuple(TextBlob(lower_caption).tokens))


            upper_tokens = [word.lower() for word in upper_cap_list ]
            lower_tokens = [word.lower() for word in lower_cap_list ]
            # upper_tokens = upper_cap_list
            # lower_tokens = lower_cap_list
            print(upper_tokens)
            print(lower_tokens)
            tokens = upper_tokens + ['<pause>'] + lower_tokens
            #print(tokens)
            counter.update(tokens)
        most_common_words = counter.most_common(max_meme_vocab)
        most_common_words, _ = zip(*most_common_words)

        token_to_id = {'<pad>': 0, '<end>': 1, '<start>': 2, '<pause>': 3, '<unk>': 4}
        most_common_words = set(most_common_words) - set(token_to_id.keys())

        token_to_vec = {}

        with open(embedding_path) as f:
            for line_num, line in enumerate(f):
                values = line.split()  # Splits on spaces.
                word = values[0]
                if word in most_common_words and not word in token_to_id:
                    vector = np.asarray(values[1:], dtype='float32')
                    token_to_vec[word] = vector

        vocab = most_common_words | set(token_to_vec.keys())
        # eta_words = ['<pad>','<start>','<pause>','<end>','<unk>']
        rand_state = np.random.RandomState(42)

        num_meta_tokens = len(token_to_id)
        embedding_matrix = rand_state.rand(len(vocab) + num_meta_tokens, 300)

        for i in range(num_meta_tokens):
            embedding_matrix[i] = i
        vocab = sorted(vocab)  # Sort for consistent ids.
        print('vocab', len(vocab))

        for i, token in enumerate(vocab):
            word_id = i + num_meta_tokens
            if token in token_to_vec:
                embedding_matrix[word_id] = token_to_vec[token]
            token_to_id[token] = word_id

        self.embedding_matrix = embedding_matrix
        self.word_to_index = token_to_id
        self.index_to_word = {id: word for word, id in self.word_to_index.items() }

        print(most_common_words, len(most_common_words))
        #
        # words = []
        # max_line_num = 2000
        # contract = list(get_stop_words('en'))
        #
        # vocab = Vocabulary()
        # meta_words = ['<pad>','<start>','<pause>','<end>','<unk>']
        # rand_state = np.random.RandomState(42)
        #
        # for index, meta in enumerate(meta_words):
        #     words.append(meta)
        #     vector = rand_state.normal(scale=0.6, size=(300, ))
        #     vocab.add_word(meta,vector)
        #
        #
        # for i,w in enumerate(contract):
        #    w = w.replace("'","")
        #    if w not in words:
        #        words.append(w)
        #        vector = rand_state.normal(scale=0.6, size=(300, ))
        #        vocab.add_word(w,vector)
        #
        #
        # with open(embedding_path) as f:
        #
        #     for line_num, line in enumerate(f):
        #         if line_num == max_line_num:
        #             break
        #
        #         values = line.split()  # Splits on spaces.
        #         word = values[0]
        #         #print(tweet_tokenizer.tokenize(word))
        #         #print([char for char in word if char not in stop])
        #         matching = ''.join([char for char in word if char not in stop])
        #         matching = ''.join(filter(str.isalpha, matching))
        #         match = (len(matching) == len(word))
        #         if match:
        #             vector = np.asarray(values[1:], dtype='float32')
        #             vocab.add_word(word,vector)
        #


        #print(vocab.word_to_index['one'],vocab.word_to_index['does'],vocab.word_to_index['not'],vocab.word_to_index['simply'])
        print(len(self.embedding_matrix))
        print(len(self.word_to_index))




def main(json, vocab_path):
    embedding_path = '/Users/mcassettix/github/final_project/project_kojak' + '/data/' + 'glove.42B.300d.txt'
    vocab = Vocabulary(json,embedding_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)



if __name__ == '__main__':
    current_dir = os.getcwd()
    json = current_dir + '/captions.json'
    vocab_path = current_dir + '/vocab.pkl'
    main(json, vocab_path)
