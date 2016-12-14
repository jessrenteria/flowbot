"""Module for preprocessing Cornell dialog data.
"""

import os
import pickle
import nltk
from tqdm import tqdm

_LINES = 'data/corpus/movie_lines.txt'
_CONVERSATIONS = 'data/corpus/movie_conversations.txt'
_PREPROCESSED_LINES = 'data/preprocessed/lines.pkl'
_PREPROCESSED_CONVERSATIONS = 'data/preprocessed/conversations.pkl'

_MAX_LENGTH = 50

class Preprocessor:
    """Class for parsing and preprocessing dialogs.
    """

    def __init__(self):
        self._lines, self._token2id, self._id2token = self._preprocess_lines()
        self._conversations = self._preprocess_conversations()

    def _preprocess_lines(self):
        if os.path.exists(_PREPROCESSED_LINES):
            with open(_PREPROCESSED_LINES, 'rb') as f:
                data = pickle.load(f)
                return data['lines'], data['token2id'], data['id2token']
        else:
            lines = {}
            token2id = {}
            id2token = {}
            next_id = 0

            def getId(token):
                nonlocal next_id
                if token in token2id:
                    return token2id[token]
                else:
                    token2id[token] = next_id
                    id2token[next_id] = token
                    next_id += 1

            for token in ['<start>', '<end>', '<pad>']:
                getId(token)

            with open(_LINES, 'r', encoding='iso-8859-1') as f:
                for line in tqdm(f):
                    line = line.split('+++$+++')
                    tokenIds = nltk.word_tokenize(line[-1])
                    if len(tokenIds) <= _MAX_LENGTH - 1:
                        lines[line[0].strip()] = list(map(getId, tokenIds))

            with open(_PREPROCESSED_LINES, 'wb') as f:
                data = {
                    'lines' : lines,
                    'token2id': token2id,
                    'id2token': id2token
                    }
                pickle.dump(data, f, -1)

            return lines, token2id, id2token

    def _preprocess_conversations(self):
        if os.path.exists(_PREPROCESSED_CONVERSATIONS):
            with open(_PREPROCESSED_CONVERSATIONS, 'rb') as f:
                return pickle.load(f)
        else:
            conversations = []

            def valid_conversation(c):
                c1, c2 = c
                return c1 in self._lines and c2 in self._lines

            with open(_CONVERSATIONS, 'r', encoding='iso-8859-1') as f:
                for line in tqdm(f):
                    conversation = line.split('+++$+++')[-1].strip()[2:-2].split("', '")
                    candidates = zip(conversation[:-1], conversation[1:])
                    candidates = list(filter(valid_conversation, candidates))
                    conversations += candidates
            with open(_PREPROCESSED_CONVERSATIONS, 'wb') as f:
                pickle.dump(conversations, f, -1)
            return conversations

    def get_data(self):
        return self._lines, self._conversations

    def decode(self, lst):
        return ' '.join(map(lambda x: self._id2token[x], lst))
