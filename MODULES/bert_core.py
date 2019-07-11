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
                 cuda: bool = True):
        # -------- CODE -------------------------------------------------------
        self._w2v_model = w2v_model
        self._vocabulary_size = vocabulary_size
        self._batch_size = batch_size
        self._latent_space_dimension = latent_space_dimension
        self._max_sequence_length = max_sequence_length
        self._max_grad_norm = max_grad_norm
        self._vector_embedding_dimension = w2v_model.vector_size
        self._learning_rate = learning_rate

        # define tensor flow variables
        self.global_step = tf.get_variable(
            'global_step', shape=[], trainable=False, initializer=tf.initializers.zeros())

        # we will train special embeddings for <bos> == beginning of sentence = 2
        # <eos> == end of sentence  is 3, <UNK> == unknown is 1 and <FILL> is 0
        # and it is the filling character.
        special_embeddings = tf.get_variable('special_embeddings',
                                             shape=[4, self._vector_embedding_dimension],
                                             initializer=tf.initializers.random_uniform(
                                                 -np.sqrt(3), np.sqrt(3)),
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


class Bert_inference():
    """
    this class provide the bert inference of the trained bert model
    """
    pass
