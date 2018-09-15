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
embed_size = 300
hidden_size = 300
batch_size = 64
num_layers = 3
max_seq_length = 10
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
    encoder_path = model_path +  '/encoder-100-1.ckpt'
    decoder_path = model_path +  '/decoder-100-1.ckpt'
    # full_model = model_path + '/full_model.pt'
    full_model = model_path + '/nick_model_3.pt'
    image_path = current_dir + '/image_resized/' + 'grumpy-cat.jpg'
    #image_path = current_dir + '/image_resized/' + 'forever-alone.jpg'
    meta_tokens = ['<pad>','<start>','<pause>','<unk>']
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406),
        #                      (0.229, 0.224, 0.225))
    ])

    # build the models
    # encoder = EncoderCNN(embed_size)
    # length = len(vocab)
    # decoder = DecoderRNN(embed_size, hidden_size, length, vocab.embedding_matrix, num_layers, max_seq_length)
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    #
    # encoder_state = torch.load(encoder_path)
    # decoder_state = torch.load(decoder_path)
    # decoder.load_state_dict(decoder_state)
    # encoder.load_state_dict(encoder_state)
    encoder, decoder = torch.load(full_model)
    encoder.to(device)
    decoder.to(device)

    if device == 'cpu':
        encoder.eval()
        decoder.eval()
    else:
        encoder.eval()
        decoder.eval()



    image_tensor = load_image(image_path, transform).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        encoder.eval()
        print('image', image_tensor)
        print('image shape', image_tensor.shape)
        feature = encoder(image_tensor)
        print('feature', feature)
    # feature = torch.zeros_like(feature)

    seed = ['<start>']
    seed = [vocab.word_to_index[word] for word in seed]
    # seed = [vocab.word_to_index['super']]
    print('i to w', vocab.index_to_word)
    print('w to i', vocab.word_to_index)

    with torch.no_grad():
        encoder.eval()
        decoder.eval()


        # X = ['<pad>', '<start>', 'super', 'rad', '<pause>', 'aadvark']
        # X = ['<pad>', '<start>']
        # X = [vocab.word_to_index[word] for word in X]
        # X = torch.LongTensor([X]).to(device)
        # y = decoder(feature, X, [X.size(1)])
        # print('X', X)
        # print('raw_y', y)
        # y = y.argmax(dim=1)
        # print('y', y)
        # print('y words', [vocab.index_to_word[id.item()] for id in y])
        # raise Exception()


        # decoder.recur_state = decoder.init_recur_state(1)
        sampled_ids = decoder.greedy(feature, seed)
        print('sampled ids', sampled_ids)
        print('sampled words', [vocab.index_to_word[id] for id in sampled_ids])
        #sampled_ids = sampled_ids[0].detach().cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        sampled_caption = []

        for word_id in sampled_ids:
            word = vocab.index_to_word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
            # if word in meta_tokens:
            #     continue
            #
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        print("greedy algo:", sentence)
    # for _ in range(1):
    #     sampled_ids = decoder.beam_search(feature)
    #     #sampled_ids = sampled_ids[0].detach().cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #     sampled_caption = []
    #     for seq, score in sampled_ids:
    #         [print(vocab.index_to_word[s]) for s in seq]
    #     #for word_id in sampled_ids:
    #     #    word = vocab.index_to_word[word_id]
    #     #    if word == '<end>':
        #        break
        #    if word in meta_tokens:
        #        continue
        #    sampled_caption.append(word)
        #sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        # print(sentence)
