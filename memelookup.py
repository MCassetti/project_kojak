import os
from collections import defaultdict
import json
import sys

class MEME:
    def __init__(self, caption_file=None):
           # load dataset
        self.dataset = dict()
        self.caps = dict()
        self.imgs = dict()
        self.imgToCaps = defaultdict(list)
        if not caption_file == None:
            print('loading captions into memory...')
            dataset = json.load(open(caption_file, 'r'))
            print(dataset)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        caps = {}
        imgs = {}
        imgToCaps = defaultdict(list)

        #if 'captions' in self.dataset:
        for caps in self.dataset.keys():
            print(caps)
            imgToCaps[caps['image_id']].append(caps)
            caps[caps['id']] = caps

        #if 'images' in self.dataset:
        for img in self.dataset.values():
            print(img)
            imgs[img['id']] = img

                # create class members
        self.caps = caps
        self.imgToCaps = imgToCaps
        self.imgs = imgs

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))
