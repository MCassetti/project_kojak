import os
from collections import defaultdict
import sys

class MEME:
    def __init__(self, caption_file=None):


    def createIndex(self):
        # create index
        print('creating index...')
        caps = {}
        imgs = {}
        imgToCaps = defaultdict(list)
        if 'captions' in self.dataset:
            for caps in self.dataset['captions']:
                imgToCaps[caps['image_id']].append(caps)
                caps[caps['id']] = caps

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
