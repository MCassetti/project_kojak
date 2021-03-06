from cnnLSTMmodel import EncoderCNN, DecoderRNN
import torch
import torch.nn as nn
import pickle
import os
import json
import timeit
import nltk
import numpy as np
from PIL import Image
from textblob import TextBlob
from memelookup import MEME
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from meme_vocabulary import Vocabulary
import string
from nltk.tokenize.casual import TweetTokenizer
from sklearn.metrics import accuracy_score
tweet_tokenizer = TweetTokenizer()


""" This is main driver for training the image/caption dataset"""
""" This is my implementation of pytorch's image captioning encoderCNN and decoderRNN"""

current_dir = os.getcwd()
data_dir = '/data/'
image_dir = '/image_resized/'
model_dir = '/models/'
vocab_path = current_dir + '/vocab.pkl'
caption_path = current_dir + data_dir + 'captions.json'
image_path = current_dir + image_dir
model_path = current_dir + model_dir
embed_size = 300
hidden_size = embed_size
batch_size = 1024
num_workers = 2
num_layers = 3
num_epochs = 10
learning_rate = 0.01
crop_size = 224
save_step = 1
log_step = 5
shuffle = True



class memeDataset(DataLoader):
    """MEME Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):

        self.image_path = image_path
        self.meme = MEME(json)
        self.ids = list(self.meme.caps.keys())
        self.vocab = vocab
        self.transform = transform
        self.root = root

    def __getitem__(self,index):
        """Returns image and data caption pair"""
        meme = self.meme
        vocab = self.vocab
        cap_id = self.ids[index]
        caption = meme.caps[cap_id]['caption']
        img_id = meme.caps[cap_id]['image_id']
        path = meme.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        vocab = self.vocab
        stop = list(string.punctuation)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        caption = caption.replace("'","")
        caption = caption.split(' <pause> ')
        upper_caption = caption[0]
        lower_caption = caption[-1]

        upper_cap_list = list(tuple(TextBlob(upper_caption).tokens))
        lower_cap_list = list(tuple(TextBlob(lower_caption).tokens))


        upper_tokens = [word.lower() for word in upper_cap_list if word in vocab.word_to_index ]
        lower_tokens = [word.lower() for word in lower_cap_list if word in vocab.word_to_index ]

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in upper_tokens])
        caption.append(vocab('<pause>'))
        caption.extend([vocab(token) for token in lower_tokens])
        caption.append(vocab('<end>'))
        #print(caption)
        captions = [cap for cap in caption]
        target = torch.Tensor(captions)

        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Returns:
        images: torch tensor of shape (batch_size, 3, 300, 300).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        embeddings: torch tensor shape of (batch_size, word embedding dim (300)).
    """
    # Sort a data list by caption length (descending order).

    info_dump = f"""
       The data is of type {type(data)}
       The length of the data is {len(data)}
       The length of the first element is {len(data[0])}
    """
    #print(info_dump, flush=True)
    # for index, (image,caption,embeddings) in enumerate(data):
    #
    #     print(f"image:{image},image size:{image.size()}")
    #     print(f"caption:{caption},captionsize: {caption.size()}")
    #     print(f"embeddings:{embeddings}, embedding_size:{embeddings.size()}")

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        targets[i, :len(cap)] = cap

    return images, targets, lengths


if __name__ == '__main__':
    # configure the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # open image and caption files
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(vocab)
    json = 'captions.json'

    # create DataLoader from my meme dataset
    # my images: a tensor of shape (batch_size, 3, crop_size, crop_size)
    # my captions: a tensor of shape (batch_size, padded_length)
    # my lengths: a list indicating valid length for each caption. length is (batch_size).
    # transform = transforms.Compose([
    #     transforms.RandomCrop(crop_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))
    # ])

    transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406),
        #                      (0.229, 0.224, 0.225))
    ])

    memedata = memeDataset(root=image_path,
                           json=json,
                           vocab=vocab,
                           transform=transform)

    data_loader = DataLoader(dataset=memedata,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)



    total_step = len(data_loader)


    # build models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), vocab.embedding_matrix, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            learning_rate = learning_rate/5
        optimizer = torch.optim.Adam(params,lr=learning_rate) # prefered for computer vision problems, Adam realizes the benefits of both AdaGrad and RMSProp.
        epoch_start = timeit.timeit()
        for i, (images, captions, lengths) in enumerate(data_loader):

            encoder.eval()
            decoder.train()
            optimizer.zero_grad()
            minibatch_start = timeit.timeit()
            X_captions = captions[:, :-1]
            X_captions = X_captions.to(device)
            y_captions = captions[:, 1:]
            y_captions = y_captions.to(device)

            # Set mini-batch dataset

            images = images.to(device)
            captions = captions.to(device)
            lengths = [seq_len - 1 for seq_len in lengths]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, X_captions, lengths)
            targets = pack_padded_sequence(y_captions, lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)
            acc = accuracy_score(targets, outputs.argmax(-1))
            print('epoch acc', acc)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info

            if i % log_step == 0:
                minibatch_end = timeit.timeit()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                print('Approx time per logstep [{}]'.format((minibatch_start - minibatch_end)))
            # Save the model checkpoints
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

        epoch_end = timeit.timeit()
        if i % log_step == 0:
            print('Approx time per epoch {}'.format((epoch_start - epoch_end)))
    print('fml')
    torch.save((encoder, decoder), os.path.join(model_path, 'full_model.pt'))
