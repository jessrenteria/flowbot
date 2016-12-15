Jess Renteria (jvr35)

Entire codebase is written in Python 3.5.x
with the following dependencies:
    - Tensorflow 0.12
    - tqdm
    - nltk

You must download and unzip the following corpus:
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
You can specify a relative path from the top-level package to
retrieve these files.

Many options can be configured in config.ini
(naming conventions in the file are self-documenting).
In general any directories in the config.ini must exist
before running the code, otherwise there may be an error.

Train the network by running

    python3 run.py --train

Note: training a decent model takes many many hours and
requires a good choice of hyperparameters.

Test the network by running

    python3 run.py

This opens an interactive session with live chatting.
