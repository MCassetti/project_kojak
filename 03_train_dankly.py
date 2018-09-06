import cnnLSTMmodel as model
import torch
import torch.nn as nn
import pickle
import os
from PIL import Image
from memelookup import MEME
from torch.utils.data import Dataset, DataLoader, pack_padded_sequence, PackedSequence
""" This is main driver for training the image/caption dataset"""
""" This is my implementation of pytorch's image captioning encoderCNN and decoderRNN"""

current_dir = os.getcwd()
data_dir = '/data/'
image_dir = '/images/resized/'
model_dir = '/models/'
vocab_path = current_dir + data_dir + 'vocab.pkl'
ids_path = current_dir + data_dir + 'ids.pkl'
caption_path = current_dir + data_dir + 'captions.json'
image_path = current_dir + image_dir
model_path = current_dir + model_dir
embed_size = 256
hidden_size = 512
batch_size = 128
num_workers = 2
num_layers = 2
num_epochs = 5
learning_rate = 0.001
crop_size = 224
save_step = 100
log_step = 10

class memeDataset(data.Dataset):
    """MEME Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, ids, transform=None):

        self.image_path = image_path
        self.meme = MEME(json)
        self.ids = list(self.meme.caps.keys())
        self.vocab = vocab
        self.transform = transform


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
        if self.transform is not None:
            image = self.transform(image)
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)



if __name__ == '__main__':
    # configure the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # open image and caption files
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    with open(ids_file,'rb') as f:
        ids = pickle.load(f)

    # create DataLoader from my meme dataset
    # my images: a tensor of shape (batch_size, 3, crop_size, crop_size)
    # my captions: a tensor of shape (batch_size, padded_length)
    # my lengths: a list indicating valid length for each caption. length is (batch_size).
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    memedata = memeDataset(image_path=image_path,
                           json=json,
                           vocab=vocab,
                           ids=ids,
                           transform=transform)

    data_loader = DataLoader(dataset=memedata,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    total_step = len(data_loader)
    # build models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate) # prefered for computer vision problems, Adam realizes the benefits of both AdaGrad and RMSProp.

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            ## Make this your own...add a visualizer
            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
