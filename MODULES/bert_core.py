import tensorflow as tf
from .bert_utils import tf_record_batch_iterator as rec_batch
import numpy as np


class Bert_train():
    """
    Parameters
    ----------
    path_to_tf_records:str
        is the full path to the training data already dumped as tf records
    w2v_model: 'loaded_w2v_model'
        is a loaded w2v_model
    vocabulary_size:int

    latent_space_dimension:int
        is the size of the latent embedding space, default is 512
    max_sequence_length:int
        is the maximum length of the sequence, must be synchronized with
        the data dump of tf records (we had 40 of true words + 2 (eos and bos?))

    learning_rate:int

    max_grad_normLfloat

    optimizer:str
        string with the name of optimizer or None, if None then we will use Adam

    cuda:bool = True
    we place to cuda if possible

    Notes
    -----
    ::time_major must be false

    """

    def __init__(self,
                 path_to_tf_records: str,
                 w2v_model: 'loaded_w2v_model',
                 vocabulary_size: int = 20000,
                 batch_size: int = 128,
                 latent_space_dimension: int = 512,
                 max_sequence_length: int = 40,
                 learning_rate: float = 1e-3,
                 max_grad_norm: float = 10,
                 optimizer: str = None,
                 number_of_rnn_in_layers=1,
                 cuda: bool = True,
                 time_major: bool = True):
        # -------- CODE -------------------------------------------------------
        self._w2v_model = w2v_model
        self._vocabulary_size = vocabulary_size
        self._batch_size = batch_size
        self._latent_space_dimension = latent_space_dimension
        self._max_sequence_length = max_sequence_length
        self._max_grad_norm = max_grad_norm
        self._vector_embedding_dimension = w2v_model.vector_size
        self._learning_rate = learning_rate
        self._number_rnn_in_layers = number_of_rnn_in_layers,
        self._cuda = cuda
        self._time_major = time_major

        # define tensor flow variables
        self.global_step = tf.get_variable(
            'global_step', shape=[], trainable=False, initializer=tf.initializers.zeros())

        # we will train special embeddings for <bos> == beginning of sentence = 2
        # <eos> == end of sentence  is 3, <UNK> == unknown is 1 and <FILL> is 0
        #
        # !!!
        # ALSO VERY IMPORTANT, THERE IS ONE MORE TOKEN
        # IT IS THE MASK TOKEN <MASK> is 4.
        # and it is the filling character.

        # below the 5 are those 5 tokens and we have sqrt(5-1)
        special_embeddings = tf.get_variable('special_embeddings',
                                             shape=[5, self._vector_embedding_dimension],
                                             initializer=tf.initializers.random_uniform(
                                                 -np.sqrt(4), np.sqrt(4)),
                                             trainable=True)

        # here get the embeddings from normal word 2 vec model
        # TO BE DONE: what follows works only if we take only first
        # `vocabulary_size` words, in the future we will do the
        # statistics with respect to the corpus and then we will take the first
        # `vocabulary_size` number of most used words
        # also by default we will train them, this might not be necessary
        # but expect that this might produce better model.
        word_embeddings = tf.get_variable('word_embeddings', shape=[
                                          vocabulary_size, self._vector_embedding_dimension],
                                          initializer=tf.initializers.constant(
                                              w2v_model.vectors[:vocabulary_size]),
                                          trainable=True)

        # ok here we are producing the actual embeddings
        self._embeddings = tf.concat([special_embeddings, word_embeddings], 0)

        # ok now we need to encode the sentence and at every step remember the
        # embedding of every word as produce of encoder
        # ------- THE OUTPUT LAYERS ----------
        #
        # ??? I am not sure about this ??? check later
        self._output_layer = tf.layers.Dense(vocabulary_size, name='output_layer')

        self._output_layer.build(latent_space_dimension)
        # ------------------------------

    def _bert_sentence_encoder(self,
                               one_data_batch_iter: 'tf record batch iter'):
        """
        ADD DOCSTRING
        """
        # -- STEP 1 -- GET train_inputs AND forward_labels FROM ONE BATCH
        # ITERATOR

        # get train_inputs and fw_labels
        train_inputs, fw_labels = tf.unstack(one_data_batch_iter, num=2)

        # -- STEP 2 -- SIMILARLY TO SKIP THOUGHT, HERE ENCODE THE train_inputs
        # BUT UNLIKE THE skip though, HERE YOU MUST RETURN ALL EMBEDDINGS
        # YOU ARE GETTING AS YOU ARE EMBEDDING THE TRAIN INPUTS
        self._encoded_sequences = self._encode_sequences(train_inputs)
        raise NotImplementedError()

    def _encode_sequences(self, train_inputs: 'tf tensor'):
        """
        ADD DOCSTRING
        """

        # -- STEP 1 -- GET INITIAL EMBEDDINGS (think about w2v embedding of sequences)
        _encoder_in = self._get_embedding(train_inputs)

        # -- STEP 2 -- USE THE GRU UNIDIRECTIONAL RNN TO CREATE CAUSALLY RELATED EMBEDDINGS
        #
        # I NEED TO RETURN THE FULL EMBEDDING TENSOR
        # FOLLOW SUGGESTION OF THIS POST
        # https://stackoverflow.com/questions/49183538/simple-example-of-cudnngru-based-rnn-implementation-in-tensorflow
        #
        # ALSO CUDA GRU EXPECTS THE TIME MAJOR!!!!
        #
        # WHAT FOLLOWS DOES NOT WORK FOR US I GUESS
        if self._cuda:
            # NOTE:
            # - the result of the `tf.contrib.cudnn_rnn.CudnnGRU` is a touple
            # - first part of that tuple is the tensor of the shape:
            # IF TIME  MAJORIZED: (sequence_length, batch_size, 2*_latent_space_dimension), if you use biderictional
            # OR (sequence_length, batch_size, _latent_space_dimension) if you use unidirectional
            #
            # - the second part of that touple is THE TUPLE of the type ('final output',)
            # the `final output` of the rnn
            # and it has shape (2, batch_size, _latent_space_dimension) if `biderictional`,
            # where 2 means == forward cell result, and backward cell result
            # or it has just shape (1, batch_size, _latent_space_dimension) if
            # 'unidirectional'
            #
            # Note, surprisingly there is not enough written in the original section of
            # the function tf.contrib.cudnn_rnn.CudnnGRU
            # it is better to read the return section in this
            # https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
            #
            # also note, there is still unsolved problem in:
            # https://github.com/tensorflow/tensorflow/issues/16042
            # basically the problem was the with the tf.contrib.cudnn_rnn.CudnnGRU
            # if you are using the following params : input_mode='skip_input', direction='bidirectional'
            #
            # note, in our case we are taking the part [0] because we want all results
            # and we want to put the self-attention mechanism
            _encoder_out = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=self._number_rnn_in_layers,
                                                         num_units=self._latent_space_dimension,
                                                         direction='unidirectional')(_encoder_in)[0]
        else:
            # ok I am doing just unidirectional, so I need only th forward cell
            _fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._latent_space_dimension)

            _sequence_length = tf.reduce_sum(tf.sign(train_inputs),
                                             reduction_indices=int(not self._time_major))

            _encoder_out = tf.nn.bidirectional_dynamic_rnn(_fw_cell,
                                                           _encoder_in, sequence_length=_sequence_length,
                                                           dtype=tf.float32, time_major=self.time_major)

            # TO DO !
            # here synchronize the api of the else branch with the api that you got from
            # the cuda branch, because current _encoder_out !=api=! _encoder_out (in cuda)

    def _get_embedding(self, query: 'tf tensor'):
        """
        ADD DOCSTRING
        """
        return tf.nn.embedding_lookup(self._embeddings, query)


class Bert_inference():
    """
    this class provide the bert inference of the trained bert model
    """
    pass
