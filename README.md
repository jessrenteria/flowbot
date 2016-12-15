# flowbot, a Tensorflow chatbot

flowbot is a Tensorflow chatbot using seq2seq at its core.

flowbot uses an encoder RNN to compute compress a sequence
of words into a *thought vector*, which is then decompressed
by a decoder RNN.

# Running flowbot

You can start an interactive session by running

    python3 run.py

or start a training session by running

    python3 run.py --train

Configuration options are encapsulated in config.ini, and are
mostly self-explanatory.

# Dependencies

- Python 3.5.x
- Tensorflow 0.12
- tqdm
- nltk

I trained it using the Cornell Movie-Dialogs Corpus, located
[here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
This corpus must be downloaded in order to train the network.

This implementation was heavily inspired by
[Concylicultor's DeepQA](https://github.com/Conchylicultor/DeepQA)

I hope to see great results by using the RNN cell my
research group developed, DizzyRNN, described
[here](https://arxiv.org/abs/1612.04035)

