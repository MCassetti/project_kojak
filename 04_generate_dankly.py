import torch
from PIL import Image
import pickle
import os
from torchvision import transforms
from cnnLSTMmodel import EncoderCNN, DecoderRNN
import numpy as np
import matplotlib.pyplot as plt
from meme_vocabulary import Vocabulary
from collections import OrderedDict
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 300
hidden_size = 300
batch_size = 64
num_layers = 3
max_seq_length = 100
crop_size = 224

# def load_state_dicts(state, model):
#     state_dict = state['state_dict']
#     new_state_dict = OrderedDict()
#     for key, value in state_dict.items():
#         #name = key
#         name = key[7:] # remove `module.`
#         new_state_dict[key] = value
#     # load params
#     return model.load_state_dict(state['state_dict'])

def load_image(image_path,transform):
    image = Image.open(image_path).convert('RGB')
    # image.resize([224, 224], Image.ANTIALIAS)
    # image = image.resize([224, 224], Image.LANCZOS)
    # image.show()
    if transform is not None:
        image = transform(image)
        # print("what is the size of this shit 1:", image)
    return image.to(device)
        # print("what is the size of this shit:",image.size())
        # image = torch.stack(image, 0)
        #image = transform(image).unsqueeze(0)
    # return image

if __name__ == '__main__':

    current_dir = os.getcwd()
    vocab_path = current_dir + '/vocab.pkl'
    model_path = current_dir + '/models/'
    full_model = model_path + '/full_model.pt'
    image_path_dir = current_dir + '/image_resized/'
    images = ['yo-dawg.jpg', 'captain-picard.jpg','grumpy-cat.jpg','success-kid_first.jpg','what-if-i-told-you-matrix-morpheus.jpg',
    'chemistry-cat.jpg','futurama-fry.jpg','image_from_ios.jpg']

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406),
        #                      (0.229, 0.224, 0.225))
    ])

    # build the models
    encoder, decoder = torch.load(full_model)
    encoder.to(device)
    decoder.to(device)

    if device == 'cpu':
        encoder.eval()
        decoder.eval()
    else:
        encoder.eval()
        decoder.eval()


    seeds = [['<start>','yo','dawg'],['<start>','why','the'],['<start>'],['<start>'],['<start>','what','if','i','told'],['<start>'],['<start>','not','sure'],['<start>']]

    for index, seed in enumerate(seeds):
        image = images[index]

        image_path = image_path_dir + image
        image_tensor = load_image(image_path, transform).to(device)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            encoder.eval()
            feature = encoder(image_tensor)
        #print(seed,image_path)
        seed = [vocab.word_to_index[word] for word in seed]
        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            three_tokens = [3,2,1]
            for _ in range(1):
                sampled_caption = []
                sampled_ids = decoder.greedy(feature, seed)

                if set(sampled_ids) == set(three_tokens):
                    sampled_ids = decoder.greedy(feature, seed)

                for word_id in sampled_ids:
                    word = vocab.index_to_word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break

                sentence = ' '.join(sampled_caption)
                #print(f"image: {image}, greedy algo: {sentence}")

            for _ in range(200):
                sampled_caption = []
                sampled_ids = decoder.softmax_probs(feature, seed)
                if set(sampled_ids) == set(three_tokens):
                    print(set(sampled_ids), set(three_tokens))
                    sampled_ids = decoder.softmax_probs(feature, seed)

                for word_id in sampled_ids:
                    word = vocab.index_to_word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break

                sentence = ' '.join(sampled_caption)
                print(f"image: {image}, softmax: {sentence}",flush=True)

            for _ in range(1):
                top_n = 10
                #print(top_n)
                sampled_ids = decoder.beam_search(feature, seed, top_n)
                #print(sampled_ids)
                sampled_caption = []

                for word_id in sampled_ids:
                    word = vocab.index_to_word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break

                sentence = ' '.join(sampled_caption)


                #print(f"image: {image}, beam search:{sentence}",flush=True)
