import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EncoderCNN(nn.Module):
    def __init__(self,embed_size):
        """ """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] #delete the last fc layer
        self.resnet = nn.Sequential(*modules) #pytorch's method of stringing together models
        self.linear = nn.Linear(resnet.fc.in_features,embed_size)
        self.bn = nn.BatchNorm1d(embed_size,momentum=0.01) #used to combat nonlinearities in CNN

    def forward(self,images):
        """Image feature extraction"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0),-1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size, embedding_matrx, num_layers,max_seq_length = 20):
        """Build layers and set hyper params"""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.drop = nn.Dropout(p=0.5, inplace=True)

    def forward(self, features, captions, embeddings, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        #outputs, _ = pack_padded_sequence(outputs, lengths, batch_first=True)
        outputs = self.drop(outputs)
        return outputs

    def greedy(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            tmpprobs = F.softmax(outputs.view(-1))
            probs = tmpprobs/sum(tmpprobs)
            probs = probs.detach().cpu().numpy()
            outputs_flat = outputs.view(1,-1).detach().cpu().numpy()
            index = np.random.choice(len(outputs.view(-1)) ,p=probs)
            #print(index, type(index), list(outputs_flat)[index], type(outputs_flat))
            predicted_max = outputs.max(dim=1)[1]                        # predicted: (batch_size)
            predicted = torch.tensor([index], dtype=torch.long).to(device)
            print(predicted.size(), predicted, predicted_max)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
