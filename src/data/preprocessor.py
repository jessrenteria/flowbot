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
        if (self._config['reuse_data'] and
                os.path.exists(self._config['preprocessed_lines'])):
            with open(self._config['preprocessed_lines'], 'rb') as f:
                data = pickle.load(f)
                return data['lines'], data['token2id'], data['id2token']
        else:
            print('Preprocessing lines...')
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
                    return 0

            for token in ['<unk>', '<start>', '<end>', '<pad>']:
                getId(token)

            max_length = max(self._config['encoder_length'],
                    self._config['decoder_length'])

            with open(self._config['lines'], 'r', encoding='iso-8859-1') as f:
                for line in tqdm(f):
                    line = line.split('+++$+++')
                    tokens = nltk.word_tokenize(line[-1])
                    if len(tokens) <= max_length:
                        tokens = map(lambda x: x.lower(), tokens)
                        lines[line[0].strip()] = list(map(getId, tokens))

            with open(self._config['preprocessed_lines'], 'wb') as f:
                data = {
                    'lines' : lines,
                    'token2id': token2id,
                    'id2token': id2token
                    }
                pickle.dump(data, f, -1)

            return lines, token2id, id2token

    def _preprocess_conversations(self):
        if (self._config['reuse_data'] and
                os.path.exists(self._config['preprocessed_conversations'])):
            with open(self._config['preprocessed_conversations'], 'rb') as f:
                return pickle.load(f)
        else:
            print('Preprocessing conversations...')
            conversations = []

            def valid_conversation(c):
                c1, c2 = c
                return (c1 in self._lines and
                        len(self._lines[c1]) <= self._config['encoder_length'] and
                        c2 in self._lines and
                        len(self._lines[c2]) + 2 <= self._config['decoder_length'])

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
        return self._lines, self._token2id, self._id2token, self._conversations

    def start_id(self):
        return self._token2id['<start>']

    def end_id(self):
        return self._token2id['<end>']

    def pad_id(self):
        return self._token2id['<pad>']

    def unk_id(self):
        return self._token2id['<unk>']

    def get_vocabulary_size(self):
        return len(self._token2id)

    def encode(self, s):
        def sanitize(w):
            return self._token2id[w] if w in self._token2id else self.unk_id()

        tokens = nltk.word_tokenize(s)
        if len(tokens) > self._config['encoder_length']:
            return None

        return list(map(sanitize, tokens))

    def decode(self, lst):
        return ' '.join(map(lambda x: self._id2token[x], lst))

    def decode_pretty(self, lst):
        result = []
        bad_set = set(['<start>', '<end>', '<pad>'])

        for tokenId in lst:
            token = self._id2token[tokenId[0][0]]
            if token in bad_set:
                continue
            elif token == '<unk>':
                result.append('???')
            else:
                result.append(token)

        if result == None:
            return None

        return ' '.join(result)
