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

    # def init_recur_state(self, batch_size):
    #     """
    #     Return an empty hidden state for the recurrent layer.
    #
    #     Args:
    #          batch_size (int): The number of training examples in
    #             each mini-batch.
    #
    #     Returns:
    #         (tuple): A tuple of torch tensors each with the shape
    #             `(num_recur_layers, batch_size, recur_size)`.
    #
    #     """
    #     return (torch.zeros(self.recurrent_layers, batch_size, self.recur_size).to(device),
    #             torch.zeros(self.recurrent_layers, batch_size, self.recur_size).to(device))

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)

        batch_size = captions.size(0)
        #print(embeddings.size())
        #print(features.unsqueeze(1).size())
        # print('before concat', embeddings.shape)
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # print('after concat', embeddings.shape)
        embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # print('packed', packed[0].shape)
        #print('feature size', features.shape)

        if features is not None:
            reshaped_features = features.unsqueeze(0)
            reshaped_features = reshaped_features.expand(3, reshaped_features.size(1), reshaped_features.size(2))
            reshaped_features = reshaped_features.contiguous()
            self.recur_state = (reshaped_features, reshaped_features)


        # reshaped_features = features.unsqueeze(1)
        # reshaped_features = reshaped_features.expand(reshaped_features.size(0), captions.size(1), reshaped_features.size(2))
        # reshaped_features = reshaped_features.contiguous()
        # print('reshaped_features', reshaped_features.shape)
        # print('embeddings 1', embeddings.shape)
        # embeddings = torch.cat((embeddings, reshaped_features), 2)
        # print('embeddings 2', embeddings.shape)



        hiddens, self.recur_state = self.lstm(embeddings, self.recur_state)
        # hiddens = hiddens.contiguous()
        # Flatten out the result so it has shape
        # `(batch_length * seq_length, recur_size)`.
        # print('hiddens 1', hiddens.shape)
        # hiddens = hiddens.view(-1, 300)
        # print('hiddens 2', hiddens.shape)
        outputs = self.linear(hiddens[0])
        #outputs, _ = pack_padded_sequence(outputs, lengths, batch_first=True)
        # outputs = self.drop(outputs)
        return outputs

    def greedy(self, features, seed, states=None):
        """Generate captions for given image features using greedy search."""

        sampled_ids = seed[:]
        seed = torch.LongTensor([seed]).to(device)

        last_token_pred = self(features, seed, [seed.size(1)])
        print('raw y', last_token_pred)

        #print('y_pred', last_token_pred.shape)

        last_token_pred = last_token_pred[-1:, :].argmax(-1).unsqueeze(0)
        sampled_ids.append(last_token_pred.item())
        #print('last pred', last_token_pred)

        #print('initial word', last_token_pred.shape)
        for _ in range(self.max_seq_length):
            last_token_pred = self(None, last_token_pred, [1])
            last_token_pred = last_token_pred[-1:, :].argmax(-1).unsqueeze(0)
            print('outputed word', last_token_pred.shape)
            sampled_ids.append(last_token_pred.item())


        # sampled_ids = []
        # eshaped_features = features.unsqueeze(0)
        # reshaped_features = reshaped_features.expand(3, reshaped_features.size(1), reshaped_features.size(2))
        # reshaped_features = reshaped_features.contiguous()
        # _, states = self.lstm(start, reshaped_features)
        # start = torch.LongTensor([seed])
        # for i in range(self.max_seq_length):
        #     hiddens, states = self.lstm(start, reshaped_features)          # hiddens: (batch_size, 1, hidden_size)
        #     outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
        #     tmpprobs = F.softmax(outputs.view(-1))
        #     probs = tmpprobs/sum(tmpprobs)
        #     probs = probs.detach().cpu().numpy()
        #     outputs_flat = outputs.view(1,-1).detach().cpu().numpy()
        #     index = np.random.choice(len(outputs.view(-1)) ,p=probs)
        #
        #     #if i == 0:
        #     #    index = 179
        #     #if i == 1:
        #     #    index = 37
        #     #if i == 2:
        #     #    index = 91
        #     #if i == 3:
        #     #    index = 720
        #     #print(index, type(index), list(outputs_flat)[index], type(outputs_flat))
        #     predicted_max = outputs.max(dim=1)[1]                        # predicted: (batch_size)
        #     predicted = torch.tensor([index], dtype=torch.long).to(device)
        #     #print(predicted.size(), predicted, predicted_max)
        #     sampled_ids.append(predicted)
        #     inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        #     inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        # sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    # def beam_search(self, features, states=None):
    #     """ Generate captions for a give features using beam search."""
    #     sampled_ids = []
    #     inputs = features.unsqueeze(1)
    #     data = []
    #     top_n = 100
    #     for i in range(self.max_seq_length):
    #         hiddens,sates = self.lstm(inputs,states)
    #         outputs = self.linear(hiddens.squeeze(1))
    #         tmpprobs = F.softmax(outputs.view(-1))
    #         probs = tmpprobs/sum(tmpprobs)
    #         probs = probs.detach().cpu().numpy()
    #         data.append(probs)
    #     sequences = [[list(), 1.0]]
    #     for row in data:
    #         all_candidates = []
    #         # expand each current sequence
    #         for i in range(len(sequences)):
    #             seq, score = sequences[i]
    #             for j in range(len(row)):
    #                 candidate = [seq + [j], score * -np.log(row[j])]
    #                 all_candidates.append(candidate)
    #             # order all candidates by score
    #             ordered = sorted(all_candidates, key=lambda tup:tup[1])
    #             sequences = ordered[:top_n]
    #     return sequences
