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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 256
hidden_size = 512
batch_size = 128
num_layers = 2
max_seq_length = 100


def load_state_dicts(state, model):
    state_dict = state['state_dict']
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        #name = key
        name = key[7:] # remove `module.`
        new_state_dict[key] = value
    # load params
    return model.load_state_dict(state['state_dict'])

def load_image(image_path,transform):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    #image.show()
    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

if __name__ == '__main__':

    current_dir = os.getcwd()
    vocab_path = current_dir + '/vocab.pkl'
    model_path = current_dir + '/models/'
    encoder_path = model_path +  'encoder-1-85.ckpt'
    decoder_path = model_path +  'decoder-1-85.ckpt'
    image_path = current_dir + '/image_resized/' + 'success-kid_first.jpg'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # build the models
    encoder = EncoderCNN(embed_size)
    length = len(vocab)
    decoder = DecoderRNN(embed_size, hidden_size, length, num_layers, max_seq_length)


    encoder_state = torch.load(encoder_path)
    decoder_state = torch.load(decoder_path)
    if device == 'cpu':
        encoder.eval()
        decoder.eval()
    else:
        encoder.float().eval()
        decoder.float().eval()

    image_tensor = load_image(image_path, transform).to(device)
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    sampled_caption = []

    for word_id in sampled_ids:
        word = vocab.index_to_word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
