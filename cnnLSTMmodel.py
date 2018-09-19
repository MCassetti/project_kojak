import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EncoderCNN(nn.Module):

    def __init__(self,embed_size):
        """ """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] #delete the last fc layer
        self.resnet = nn.Sequential(*modules) #pytorch's method of stringing together models
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size,momentum=0.01) #used to combat nonlinearities in CNN

    def forward(self, images):
        """Image feature extraction"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        features = self.linear(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embedding_matrix, num_layers,max_seq_length = 20):
        """Build layers and set hyper params"""
        super(DecoderRNN, self).__init__()
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.FloatTensor(embedding_matrix).to(device)
        vocab_size = embedding_matrix.size(0)
        embed_size = embedding_matrix.size(1)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight = nn.Parameter(embedding_matrix)
        self.embed.requires_grad = True
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.recurrent_layers = num_layers
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.recur_size = hidden_size


    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        batch_size = captions.size(0)
        embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True)

        if features is not None:
            reshaped_features = features.unsqueeze(0)
            reshaped_features = reshaped_features.expand(3, reshaped_features.size(1), reshaped_features.size(2))
            reshaped_features = reshaped_features.contiguous()
            self.recur_state = (reshaped_features, reshaped_features)

        hiddens, self.recur_state = self.lstm(embeddings, self.recur_state)
        outputs = self.linear(hiddens[0])

        return outputs

    def greedy(self, features, seed, states=None):
        """Generate captions for given image features using greedy search."""

        sampled_ids = seed[:]
        seed = torch.LongTensor([seed]).to(device)

        last_token_pred = self(features, seed, [seed.size(1)])
        last_token_pred = last_token_pred[-1:, :].argmax(-1).unsqueeze(0)
        sampled_ids.append(last_token_pred.item())
        for _ in range(self.max_seq_length):
            last_token_pred = self(None, last_token_pred, [1])
            last_token_pred = last_token_pred[-1:, :].argmax(-1).unsqueeze(0)
            sampled_ids.append(last_token_pred.item())
        return sampled_ids

    def softmax_probs(self, features, seed, states=None):
        sampled_ids = seed[:]
        seed = torch.LongTensor([seed]).to(device)

        last_token_pred = self(features, seed, [seed.size(1)])
        outputs = last_token_pred[-1:, :].unsqueeze(0)
        tmpprobs = F.softmax(outputs.view(-1))
        probs = tmpprobs/sum(tmpprobs)
        probs = probs.detach().cpu().numpy()
        last_token_pred = np.random.choice(len(outputs.view(-1)) ,p=probs)
        sampled_ids.append(last_token_pred)
        last_token_pred = torch.LongTensor([last_token_pred]).to(device)
        last_token_pred = last_token_pred.unsqueeze(0)
        #print('outputed word', last_token_pred.shape)


        for _ in range(self.max_seq_length):
            last_token_pred = self(None, last_token_pred, [1])
            outputs = last_token_pred[-1:, :].unsqueeze(0)
            tmpprobs = F.softmax(outputs.view(-1))
            probs = tmpprobs/sum(tmpprobs)
            probs = probs.detach().cpu().numpy()
            last_token_pred = np.random.choice(len(outputs.view(-1)) ,p=probs)
            sampled_ids.append(last_token_pred)
            last_token_pred = torch.LongTensor([last_token_pred]).to(device)
            last_token_pred = last_token_pred.unsqueeze(0)


        return sampled_ids


    def beam_search(self, features, seed, top_n, states=None):

        data = []
        sampled_ids = seed[:]
        seed = torch.LongTensor([seed]).to(device)
        last_token_pred = self(features, seed, [seed.size(1)])
        outputs = last_token_pred[-1:, :].unsqueeze(0)
        tmpprobs = F.softmax(outputs.view(-1))
        probs = tmpprobs/sum(tmpprobs)
        probs = tmpprobs
        probs = probs.detach().cpu().numpy()
        data.append(probs)
        for i in range(self.max_seq_length):
            last_token_pred = self(features, seed, [seed.size(1)])
            outputs = last_token_pred[-1:, :].unsqueeze(0)
            tmpprobs = F.softmax(outputs.view(-1))
            probs = tmpprobs
            probs = tmpprobs/sum(tmpprobs)
            probs = probs.detach().cpu().numpy()
            data.append(probs)

        #print(probs,probs.shape)
        sequences = [[list(), 1.0]]
        for row in data:
            all_candidates = []
            # expand each current sequence
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -np.log(row[j])]
                    all_candidates.append(candidate)
                # order all candidates by score
                all_candidates.sort(key=lambda tup:tup[1])

        sampled_ids = [id[0] for i,(id,prob) in enumerate(all_candidates[:top_n])]

        return sampled_ids
