"""Module for chatbot
"""
from tqdm import tqdm
import tensorflow as tf

from model.model import Model
from data.preprocessor import Preprocessor
from data.sampler import Sampler

class Flowbot:
    """Top-level class for chatbot
    """

    def __init__(self, config, testing):
        self._config = config
        self._step_num = 0
        self._checkpoint_file = self._config['checkpoint_dir'] + self._config['checkpoint_name']
        self._preprocessor = Preprocessor(self._config)
        self._sampler = Sampler(self._config, self._preprocessor)
        self._writer = tf.summary.FileWriter(self._config['tensorboard_dir'])
        self._model = Model(config, self._preprocessor, testing)
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self._load_model()

        if self._step_num == 0:
            self._writer.add_graph(self._sess.graph)

    def _load_model(self):
        latest = tf.train.latest_checkpoint(self._config['checkpoint_dir'])

        if latest == None:
            self._sess.run(tf.global_variables_initializer())
        else:
            print('Restoring session...')
            self._saver.restore(self._sess, latest)
            print('Session restored.')

    def _save_model(self):
        print('Saving model...')
        self._saver.save(self._sess, self._checkpoint_file, global_step=self._step_num)
        print('Model saved.')

    def train(self):
        try:
            for epoch in range(self._config['num_epochs']):
                print('Epoch {}'.format(epoch + 1))

                average_loss = 0
                batch_count = 0

                for batch in tqdm(self._sampler.get_epoch()):
                    ops, feed_dict = self._model.step(batch)
                    _, loss, summary = self._sess.run(ops, feed_dict=feed_dict)
                    average_loss += loss
                    batch_count += 1
                    self._writer.add_summary(summary, global_step=self._step_num)
                    self._step_num += 1

                average_loss /= batch_count
                print('Average loss: {}'.format(average_loss))

                if epoch % 2 == 0:
                    self._save_model()
        except (KeyboardInterrupt, SystemExit):
            print('Saving and exiting...')

        self._save_model()

    def interact(self):
        print('Start a conversation:')
        print(' ' * 4 + "type 'quit' to exit\n")

        while True:
            sentence = input('> ')
            print('')
            if sentence == 'quit':
                print(' ' * 4 + 'Goodbye...\n')
                break
            encoded = self._preprocessor.encode(sentence)

            if encoded == None:
                print("I don't understand sentences that long.\n")
                continue

            batch = self._sampler.get_batch([(encoded, [])], testing=True)
            ops, feed_dict = self._model.step(batch)
            outputs = self._sess.run(ops, feed_dict=feed_dict)
            decoded = self._preprocessor.decode_pretty(outputs)

            if decoded == None:
                print('<failed to decode>\n')
                continue

            print(' ' * 4 + decoded + '\n')
