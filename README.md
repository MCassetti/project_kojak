## MEME Caption Generation


#### About

This MEME captioning project was inspired by the Stanford NLP Project
entitled "DankLearning", which was done some pretty awesome results (see this paper https://web.stanford.edu/class/cs224n/reports/6909159.pdf )

I used a similar approach, only with a simplified model and also I utilized Pytorch
to build my models.

Additionally, I was able to generate new meme captions for data that it wasn't trained on,
which I believe with more work might actually be a useful application.

#### Data
Scraped MEME images and captions from https://Memegenerator.net.
This site made my life much easier as the captions are embedded in
the description of the images. See 01_scrape_dankly.py for how this was
achieved.

#### TOOLS
Python Tools:
```python3
BeautifulSoup
TextBlob
Pytorch
```
For Data loading:
Custom Meme Class used by the Dataloader in Pytorch (see meme_vocabulary.py)
Computing: Deep Learning AMI with Cuda on AWS

#### DATA PROCESSING
To have a MEME vocabulary, a custom meme_vocabulary class was created
which will add the most common meme words into a lookup table which
will be used. In my case, 5000 unique words where used.
The embedding matrix of vocab size * embedding size was created.
To make the model not have a bunch of garbage, text blob was used for tokenization.


See 02_preprocess_step1.py and 02_preprocess_step2.py. Step1 resizes the images
and step2 is for creating a captions.json file to be used to create the vocab
and also by the data loader in the training step.


#### Use of Meta Tokens to learn the "joke" syntax

Part of the secret sauce is that the bottom caption is often the punch line of the
MEME. To capture this syntactically, I used a "pause" meta token to indicate
top and bottom caption. In the future it may be useful to separate these out entirely
into two networks.

#### MEME Caption - CNN Encoder -> LSTM Decoder
The CNN Encoder is used to take images and generate a "semantic" representation
of the image, which can be directly input into an LSTM to decode into a caption.
This is a very slick approach as it begins to learn which images features produce which
captions.

To see the model training step, please refer to 03_train_dankly.py, which
instantiates my MEME class data set and trains and saves it both at incremental checkpoints.
There is some weirdness that may be hard to follow with the model.eval() and model.train()
mode, however it's just too much to talk about here. See the next section.   

### PYTORCH IS QUIRKY

I will write a separate blog post which talks about the things I found that
you need for your model to work and other weirdness that isn't well documented.


#### CHALLENGES
The Pytorch image captioning tutorial was useful for understanding how to leverage
the pre-trained GoogleNet model, however, the LSTM they used does not in fact predict
the t+1 time step, nor does it use word embeddings. I originally was unsuccessful at training my model because
it was not actually able to predict the next word in the sequence. After some refactoring
and also eliminating the random transformation, I was able to train a toy example with
two or three captions in which the accuracy was 1.0 after 100 epochs. This allowed me
to memorize the caption and prove to myself this model was indeed valid.

After this point I chose 7 MEMES to use in my model and was able to achieve 42% accuracy
after 20 epochs. The loss function was fully converged by this point after using a learning
rate that monotonically decreased every 5 epochs. In my case, 42% accuracy is not necessarily
a bad thing, as random variation was helpful in coming up with new memes in my inference step.

Additionally, my data is riddled with the best and the worst MEMEs of humanity which
means the MEMEs generated were often explicit, derogatory, or downright lame.

This delicate balance between humor, lameness, and political correctness is hard
to capture as a human and even harder to capture as a neural net.

#### FUTURE WORK
The first thing I would like to improve is the LSTM to incorporate Attention.
When the output vectors for each word prediction are aggregated, it learned where the important information is
concentrated. I believe this may help with some of the nonsensical MEMEs that were outputted.

Additionally, being able to quantify hilarity is some way to automatically recognize if a MEME is hilarious
would be useful. In this case being able to generate both the image and the caption of the MEME to
generate entirely new MEMEs may be possible.
