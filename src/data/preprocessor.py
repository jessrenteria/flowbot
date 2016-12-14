"""Module for parsing Cornell dialog data.

"""
import os
import pickle

_LINES = 'data/corpus/movie_lines.txt'
_CONVERSATIONS = 'data/corpus/movie_conversations.txt'
_PREPROCESSED_LINES = 'data/corpus/preprocessed/lines.pkl'
_PREPROCESSED_CONVERSATIONS = 'data/corpus/preprocessed/conversations.pkl'

class Preprocessor:
    def __init__(self):
        self._lines = self._preprocess_lines()
        self._conversations = self._preprocess_conversations()

    def _preprocess_lines(self):
        if os.path.exists(_PREPROCESSED_LINES):
            with open(_PREPROCESSED_LINES, 'wb') as f:
                return pickle.load(_PREPROCESSED_LINES)
        else:
            lines = {}
            with open(_LINES, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    line = line.split('+++$+++')
                    lines[line[0].strip()] = line[-1].strip()
            with open(_PREPROCESSED_LINES, 'wb') as f:
                pickle.dump(lines, f, -1)
            return lines

    def _preprocess_conversations(self):
        if os.path.exists(_PREPROCESSED_CONVERSATIONS):
            with open(_PREPROCESSED_CONVERSATIONS, 'wb') as f:
                return pickle.load(_PREPROCESSED_CONVERSATIONS)
        else:
            conversations = []
            with open(_CONVERSATIONS, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    conversation = []
                    keys = line.split('+++$+++')[-1].strip().[2:-2].split("', '")
                    for key in keys:
                        conversation.append(self._lines[key])
                    conversations.append(conversation)
            with open(_PREPROCESSED_CONVERSATIONS, 'wb') as f:
                pickle.dump(conversations, f, -1)
            return conversations

    def get_conversations(self):
        return self._conversations
