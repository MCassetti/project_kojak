import nltk
import pickle
import os
from collections import Counter
from memelookup import MEME

class Vocabulary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index = 0

    def add_word(self,word):
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

def make_vocab(json):
    meme = MEME(json)
    ids = meme.caps.keys()
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(meme.caps[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        print(caption)

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<pause>') #to deliniate top and bottom meme caption
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # build the vocab
    words = [word for word, cnt in counter.items()]
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab



def main(json, vocab_path):
    vocab = make_vocab(json)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    current_dir = os.getcwd()
    json = current_dir + '/captions.json'
    vocab_path = current_dir + '/vocab.pkl'
    main(json, vocab_path)
