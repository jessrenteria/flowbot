"""Module for sampling dialogs.
"""
import random
import configparser
import pickle
import nltk
import tensorflow as tf

from data.preprocessor import Preprocessor
from model.model import Batch

class Sampler:
    """Class for sampling dialogs.
    """

    def __init__(self, config, preprocessor):
        self._config = config
        self._preprocessor = preprocessor
        self._lines, _, _, self._conversations = preprocessor.get_data()

    def get_batch(self, examples):
        batch = Batch()

        start_id = self._preprocessor.start_id()
        end_id = self._preprocessor.end_id()
        pad_id = self._preprocessor.pad_id()

        for idx in range(len(examples)):
            example = example[idx]

            batch.encoder_inputs.append(list(reversed(example[0])))
            batch.decoder_inputs.append([start_id] + example[1] + [end_id])
            batch.decoder_targets.append(batch.decoder_inputs[1:])

            padding = self._config['encoder_length'] - len(batch.encoder_inputs[idx])
            batch.encoder_inputs[idx] = [pad_id] * padding + batch.encoder_inputs[idx]
            padding = self._config['decoder_length'] - len(batch.decoder_inputs[idx])
            batch.decoder_inputs[idx] += [pad_id] * padding
            padding += 1
            batch.decoder_targets[idx] += [pad_id] * padding
            batch.weights.append([1.0] * len(batch.decoder_targets[idx]) + [0.0] * padding)

        encoder_inputsT = []

        for idx in range(self._config['encoder_length']):
            encoder_inputT = []
            for inner in range(len(examples)):
                encoder_inputT.append(batch.encoder_inputs[inner][idx])
            encoder_inputsT.append(encoder_inputT)

        batch.encoder_inputs = encoder_inputsT

        decoder_inputsT = []
        decoder_targetsT = []
        decoder_weightsT = []

        for idx in range(self._config['decoder_length']):
            decoder_inputT = []
            decoder_targetT = []
            decoder_weightT = []
            for inner in range(len(examples)):
                decoder_inputT.append(batch.decoder_inputs[inner][idx])
                decoder_targetT.append(batch.decoder_targets[inner][idx])
                decoder_weightT.append(batch.decoder_weights[inner][idx])
            decoder_inputsT.append(decoder_inputT)
            decoder_targetsT.append(decoder_targetT)
            decoder_weightsT.append(decoder_weightT)

        batch.decoder_inputs = decoder_inputsT
        batch.decoder_targets = decoder_targetsT
        batch.decoder_weights = decoder_weightsT

        return batch

    def get_epoch(self):
        random.shuffle(self._conversations)
        num_examples = len(self._conversations)

        batches = []

        for idx in range(0, num_examples, self._config['batch_size']):
            bound = min(num_examples, idx + self._config['batch_size'])
            batches.append(self.get_batch(self._conversations[idx:bound]))

        return batches

