"""Module for creating a seq2seq model.
"""
import tensorflow as tf

class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.decoder_weights = []

class _LinearWithBiasOp:
    def __init__(self, shape, dtype=None, scope=None):
        self._scope = scope

        with tf.variable_scope('weights_' + self._scope):
            self._W = tf.get_variable('W', shape, dtype=dtype,
                    initializer=tf.random_normal_initializer())
            self._b = tf.get_variable('b', shape[1], dtype=dtype,
                    initializer=tf.constant_initializer())

    def get_params(self):
        return self._W, self._b

    def __call__(self, X):
        with tf.name_scope(self._scope):
            return tf.matmul(X, self._W) + self._b

class Model:
    """seq2seq model.
    """

    def __init__(self, config, preprocessor, testing):
        self._encoder_length = config['encoder_length']
        self._decoder_length = config['decoder_length']
        self._num_stacked = config['num_stacked']
        self._hidden_state_size = config['hidden_state_size']
        self._embedding_size = config['embedding_size']
        self._softmax_sample_size = config['softmax_sample_size']
        self._learning_rate = config['learning_rate']
        self._testing = testing

        self._preprocessor = preprocessor
        self._vocabulary_size = preprocessor.get_vocabulary_size()

        self._model = self._buildGraph()

    def _buildGraph(self):
        output_projection = _LinearWithBiasOp(
                [self._hidden_state_size, self._vocabulary_size],
                scope='OutputProjection')

        cell = tf.nn.rnn_cell.LSTMCell(self._hidden_state_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_stacked)

        with tf.name_scope('Encoder'):
            self._encoder_inputs = [tf.placeholder(tf.int32, [None, ])
                    for _ in range(self._encoder_length)]

        with tf.name_scope('Decoder'):
            self._decoder_inputs = [tf.placeholder(tf.int32, [None, ])
                    for _ in range(self._decoder_length)]
            self._decoder_targets = [tf.placeholder(tf.int32, [None, ])
                    for _ in range(self._decoder_length)]
            self._decoder_weights = [tf.placeholder(tf.float32, [None, ])
                    for _ in range(self._decoder_length)]

        outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self._encoder_inputs,
                    self._decoder_inputs,
                    cell,
                    self._vocabulary_size,
                    self._vocabulary_size,
                    embedding_size=self._embedding_size,
                    output_projection=output_projection.get_params(),
                    feed_previous=self._testing
                )

        def sampled_softmax(inputs, targets):
            W, b = output_projection.get_params()
            W = tf.transpose(W)
            targets = tf.reshape(targets, [-1, 1])

            return tf.nn.sampled_softmax_loss(W, b, inputs, targets,
                    self._softmax_sample_size, self._vocabulary_size)

        if self._testing:
            self._outputs = [output_projection(output) for output in outputs]
        else:
            with tf.name_scope('loss'):
                self._loss = tf.nn.seq2seq.sequence_loss(
                            outputs,
                            self._decoder_targets,
                            self._decoder_weights,
                            softmax_loss_function=sampled_softmax
                        )

            opt = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            self._step = opt.minimize(self._loss)

    def step(self, batch):
        feedDict = {}
        ops = []

        if self._testing:
            for idx in range(self._encoder_length):
                feedDict[self._encoder_inputs[idx]] = batch.encoder_inputs[idx]
            for idx in range(self._decoder_length):
                feedDict[self._decoder_inputs[idx]] = batch.decoder_inputs[idx]
                feedDict[self._decoder_targets[idx]] = batch.decoder_targets[idx]
                feedDict[self._decoder_weights[idx]] = batch.decoder_weights[idx]

            ops = [self._step, self._loss]
        else:
            for idx in range(self._encoder_length):
                feedDict[self._encoder_inputs[idx]] = batch.encoder_inputs[idx]

            feedDict[self._decoder_inputs[0]] = [self._preprocessor.start_id()]
            ops = [self._outputs]

        return ops, feedDict
