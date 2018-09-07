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
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        caps = {}
        imgs = {}
        imgToCaps = defaultdict(list)

        if 'captions' in self.dataset:
            for cap in self.dataset['captions']:
                imgToCaps[cap['image_id']].append(caps)
                caps[cap['id']] = cap

        if 'images' in self.dataset:
            for img in self.dataset['images']:
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

    def loadImgs(self, ids=[]):
        """
        Load caption_links with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == int:
            return [self.imgs[ids]]
        else:
            return [self.imgs[id] for id in ids]
