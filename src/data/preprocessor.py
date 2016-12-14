"""Module for preprocessing Cornell dialog data.
"""

import os
import pickle
import nltk
from tqdm import tqdm

class Preprocessor:
    """Class for parsing and preprocessing dialogs.
    """

    def __init__(self, config):
        self._config = config
        self._lines, self._token2id, self._id2token = self._preprocess_lines()
        self._conversations = self._preprocess_conversations()

    def _preprocess_lines(self):
        if os.path.exists(self._config['preprocessed_lines']):
            with open(self._config['preprocessed_lines'], 'rb') as f:
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

            with open(self._config['lines'], 'r', encoding='iso-8859-1') as f:
                for line in tqdm(f):
                    line = line.split('+++$+++')
                    tokenIds = nltk.word_tokenize(line[-1])
                    if len(tokenIds) <= self._config['max_length'] - 1
                        lines[line[0].strip()] = list(map(getId, tokenIds))

            with open(self._config['preprocessed_lines'], 'wb') as f:
                data = {
                    'lines' : lines,
                    'token2id': token2id,
                    'id2token': id2token
                    }
                pickle.dump(data, f, -1)

            return lines, token2id, id2token

    def _preprocess_conversations(self):
        if os.path.exists(self._config['preprocessed_conversations']):
            with open(self._config['preprocessed_conversations'], 'rb') as f:
                return pickle.load(f)
        else:
            conversations = []

            def valid_conversation(c):
                c1, c2 = c
                return c1 in self._lines and c2 in self._lines

            with open(self._config['conversations'], 'r', encoding='iso-8859-1') as f:
                for line in tqdm(f):
                    conversation = line.split('+++$+++')[-1].strip()[2:-2].split("', '")
                    candidates = zip(conversation[:-1], conversation[1:])
                    candidates = list(filter(valid_conversation, candidates))
                    conversations += candidates
            with open(self._config['preprocessed_conversations'], 'wb') as f:
                pickle.dump(conversations, f, -1)
            return conversations

    def get_data(self):
        return self._lines, self._conversations

    def decode(self, lst):
        return ' '.join(map(lambda x: self._id2token[x], lst))
