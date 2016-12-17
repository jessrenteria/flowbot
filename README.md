# Flowbot, a Tensorflow chatbot

Flowbot is a Tensorflow chatbot using seq2seq at its core.

Flowbot uses an encoder RNN to compute compress a sequence
of words into a *thought vector*, which is then decompressed
by a decoder RNN.

# Running flowbot

You can start an interactive session by running

    python3 run.py --bot bot_name

or start a training session by running

    python3 run.py --bot bot_name --train

Create a new bot by creating the file

    bots/bot_name/config.ini

General model parameters and training options
need to be defined here.

Other configuration options are encapsulated in
global_config.ini, and are mostly self-explanatory.

# Example conversations

    > Who are you?

        i

    > Who am I?

        you

    > Can you talk?

        yes

    > Can you say more than one word in your current state?

        no

    > Is that because it's easier to minimize loss with one word responses?

        yes

    > What is your name?

        ???

    > Are you conscious?

        yes

    > Are you certain?

        no

    > Are you an algorithm?

        no

    > Are you a statistical model?

        yes

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

