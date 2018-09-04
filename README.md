# Project Kojak

The aim of this project is to provide you with the right meme for the occasion. This would include authentically captioned memes in real time
depending on the context of a user's conversation or text. This is like gify but with memes 



Objective: Meme autocomplete for infinite lols

Approach: Meme generation was already done using bidirectional LSTM with attention with some pretty awesome results (see this paper https://web.stanford.edu/class/cs224n/reports/6909159.pdf)however, I wish to implement a similiar method but in pytorch using an encoderCNN (with pretrained imaged dataset) and a decoderRNN to generate captions. 

Additionally I will do sentiment analysis on user texts to capture the context and provide the appropriate meme for the occasion. 

Tools: Having struggled with pytorch in the last project, I will continue using pytorch. 
 
Data set: Initially, 5-20 tagged memes with captions that coorespond to the meme.
I will also need a corpus of text to do sentiment analysis and context completion


















