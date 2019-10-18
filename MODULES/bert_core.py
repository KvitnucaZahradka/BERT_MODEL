#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import bert_utils as but
from bert_utils import tf_record_batch_iterator as rec_batch
import numpy as np

from typing import Generator, Iterator
from collections import defaultdict


class BertTrain:
    """
    Parameters
    ----------
    path_to_tf_records:str
        is the full path to the training data already dumped as tf records
    w2v_model: 'loaded_w2v_model'
        is a loaded w2v_model

    vocabulary_size:int

    n_parallel_layers: int
        is the number of parallel plates the encoder and decoder lives on, default is 6
        (see: http://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model)

    latent_space_dimension:int
        is the size of the latent embedding space, default is 512
    max_sequence_length:int
        is the maximum length of the sequence, must be synchronized with
        the data dump of tf records (we had 40 of true words + 2 (eos and bos?))

    learning_rate:int

    rnn_direction:str = restricted string, either `unidirectional` or `bidirectional`

    max_grad_normLfloat

    optimizer:str
        string with the name of optimizer or None, if None then we will use Adam

    cuda:bool = True
    we place to cuda if possible

    num_heads:int
        is the number of parallel heads used in the model
        (assert attention_hidden_dim % num_heads == 0)

    mask
        is either None: in that case we are not masking anything
        or can tensor, holding indices we are masking on input
        or can be 0< float <1 in which cas it tells us what percentage of
        indices we should randomly mask in the input tensor.
        for mask; we use the <MASK> index == 4

    Notes
    -----
    ::time_major must be false

    """

    def __init__(self, path_to_tf_records: str, w2v_model: 'loaded_w2v_model', num_heads: int,
                 n_parallel_layers: int = 6, mask=None, dropout_prob: float = 0.8, dropout: str = None,
                 vocabulary_size: int = 20000, batch_size: int = 128, latent_space_dimension: int = 512,
                 max_sequence_length: int = 40, learning_rate: float = 1e-3, max_grad_norm: float = 10,
                 optimizer: str = None, number_of_rnn_in_layers: int = 1, rnn_direction:str = 'bidirectional',
                 cuda: bool = True, time_major: bool = True):

        self._N_parallel_layers = n_parallel_layers
        self._w2v_model = w2v_model
        self._vocabulary_size = vocabulary_size

        # dimensions of batch
        self._batch_size = batch_size
        self._latent_space_dimension = latent_space_dimension
        self._max_sequence_length = max_sequence_length

        self._max_grad_norm = max_grad_norm
        self._vector_embedding_dimension = w2v_model.vector_size
        self._learning_rate = learning_rate

        self._number_rnn_in_layers = number_of_rnn_in_layers,
        self._cuda = cuda
        self._time_major = time_major

        self._direction = rnn_direction
        self._mask = mask
        self._dropout_prob = dropout_prob

        self._optimizer = optimizer
        self._dropout = dropout

        # this holds the number of attention heads
        self._num_heads = num_heads

        assert latent_space_dimension % num_heads == 0

        # here you will store the individual attention layers
        self._attention_layers = defaultdict(lambda: [])

        # define tensor flow variables
        self.global_step = tf.get_variable('global_step', shape=[], trainable=False,
                                           initializer=tf.initializers.zeros())

        # we will train special embeddings for <bos> == beginning of sentence = 2
        # <eos> == end of sentence  is 3, <UNK> == unknown is 1 and <FILL> is 0
        #
        # !!!
        # ALSO VERY IMPORTANT, THERE IS ONE MORE TOKEN
        # IT IS THE MASK TOKEN <MASK> is 4.
        # and it is the filling character.

        # below the 5 are those 5 tokens and we have sqrt(5-1)
        special_embeddings = tf.get_variable('special_embeddings', shape=(5, self._vector_embedding_dimension),
                                             initializer=tf.initializers.random_uniform(-np.sqrt(4), np.sqrt(4)),
                                             trainable=True)

        # here get the embeddings from normal word 2 vec model
        # TO BE DONE: what follows works only if we take only first
        # `vocabulary_size` words, in the future we will do the
        # statistics with respect to the corpus and then we will take the first
        # `vocabulary_size` number of most used words
        # also by default we will train them, this might not be necessary
        # but expect that this might produce better model.
        word_embeddings = tf.get_variable('word_embeddings', shape=[vocabulary_size, self._vector_embedding_dimension],
                                          initializer=tf.initializers.constant(w2v_model.vectors[:vocabulary_size]),
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

    def _encoder_mask(self, input_tensor: tf.Tensor, mask_percentage: float, replacement: (int, float)) -> tf.Tensor:
        """
        This is the masking function that we will apply only during the training

        Parameters
        ----------
        input_tensor: tf.Tensor

        mask_percentage:float
            is the float between 0 and 1 and represents the percentage of

        replacement: is float or int
            represents what percentage of the input tensor you want to mask.

        Returns
        -------
        tf.Tensor

        Notes
        -----

        Examples
        --------

        """

        # -- step 0 -- create the shape
        # this holds the shape of one tensor in a batch
        # also note that this is BEFORE the time majorization
        _slice_shape = (self._latent_space_dimension, self._max_sequence_length)

        # -- step 1 -- create the function you want to apply over the batch
        _mask_application = lambda in_tensor: but.apply_random_fill_mask(in_tenso=in_tensor,
                                                                         in_tensor_shape=_slice_shape,
                                                                         replacement=replacement,
                                                                         replacement_rate=mask_percentage)

        return tf.map_fn(lambda in_tensor: _mask_application(in_tensor), input_tensor, dtype=tf.int32)

    def _bert_sentence_encoder(self, one_data_batch_iter: Iterator[tf.Tensor], batch_size: int, train: bool = True,
                               layer_name: str = 'bert_sentence_encoder', **kwargs) -> (tf.Tensor, tf.Tensor):
        """
        ADD DOCSTRING

        Parameters
        ----------
        one_data_batch_iter: iterator over the tensor

        train: bool
            is boolean variable indicating whether you are doing the training or no

        batch_size: int
            is the size of one batch, must match with the batch used in preparation of
            `one_data_batch_iter`

        **mask_percentage:float
            determines what percentage of input tensor should be masked
            -- default -- is 0.2

        **replacement: int or float
            is the replacement//masking `symbol`, can be either float or int
            -- default -- is 4, since it is a symbol for '<MASK>' token.

        Returns
        -------
            tuple
                is the tuple of tensors, encoded input and the `forward labels`.
        """
        # -- defaults --
        _mask_percentage = kwargs.get('mask_percentage', 0.2)
        _replacement = kwargs.get('replacement', 4)

        # -- STEP 1 -- GET train_inputs AND forward_labels FROM ONE BATCH
        # ITERATOR

        # get train_inputs and fw_labels
        train_inputs, fw_labels = tf.unstack(one_data_batch_iter, num=2)

        # -- STEP 2 -- APPLY THE ENCODER MASK
        # ?? only at the time of training not inference ??
        if train:
            train_inputs = self._encoder_mask(train_inputs, mask_percentage=_mask_percentage, replacement=_replacement)

        # -- STEP 2 -- SIMILARLY TO SKIP THOUGHT, HERE ENCODE THE train_inputs
        # BUT UNLIKE THE skip though, HERE YOU MUST RETURN ALL EMBEDDINGS
        # YOU ARE GETTING AS YOU ARE EMBEDDING THE TRAIN INPUTS

        # we will use the local layer, so we can do the skip connection

        # question is whether you do not need to normalize this somehow specially??? so the addition
        _encoded_sequences_positional = self._positional_encode_sequences(train_inputs)

        # -- STEP 3 -- CLONE MULTI HEAD LAYER
        # we must clone the given layer N times
        _resulting_list = []

        for _ind in range(self._N_parallel_layers):

            # ?? I am not sure about the usage of identity here ???
            _input_x = tf.identity(_encoded_sequences_positional, name="{}_{}".format(layer_name, _ind))

            # -- STEP 3 -- DO THE MULTI-HEAD ATTENTION
            _encoded_sequences_multi_head = \
                self._multi_head_attention(name_of_attention='{}_{}'.format(layer_name, _ind), query=_input_x,
                                           key=_input_x, value=_input_x, num_heads=self._num_heads,
                                           batch_size=batch_size, d_model=self._latent_space_dimension)

            # -- STEP 4 -- APPLY LAYER NORM TO MULTI HEAD SUBLAYER
            _encoded_sequences_multi_head = tf.contrib.layers.layer_norm(_encoded_sequences_multi_head)

            # -- STEP 5 -- APPLY DROPOUT
            _encoded_sequences_multi_head = tf.nn.dropout(_encoded_sequences_multi_head, keep_prob=self._dropout_prob)

            # -- STEP 6 -- ADD THE RESIDUAL CONNECTION
            _encoded_sequences_multi_head = tf.add(_encoded_sequences_positional, _encoded_sequences_multi_head)

            # -- STEP 7 -- ADD RELU FULLY CONNECTED NN (d_model::512 --> d_inner:: 4*512 = 2048)
            # and then add linear 2048 --> 512
            # maybe there is a better way how to do that; technically this should be
            # `Position-wise Feed-Forward Networks` as in :
            # http://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model
            # but for now this should be ok
            # we fix the inner dimension to be 4*hidden_dim = 512
            _encoded_dense = tf.layers.dense(_encoded_sequences_multi_head, self._latent_space_dimension*4,
                                             activation=tf.nn.relu, kernel_initializer='glorot_uniform_initializer')

            # ?? maybe only at the time of training ??
            _encoded_dense = tf.layers.dense(_encoded_dense, self._latent_space_dimension, activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())

            # -- STEP 8 -- REPEAT NORM AND DROPOUT AND SKIP CONNECTION LAYER
            _encoded = tf.contrib.layers.layer_norm(_encoded_dense)

            # TO DO
            #?? should not this be only in the case of training???
            _encoded = tf.nn.dropout(_encoded, keep_prob=self._dropout_prob)

            _encoded = tf.add(_encoded_dense, _encoded)

            # -- STEP 9 -- PUT INTO THE CONTAINER
            _resulting_list += [_encoded]

        # -- STEP 10 -- put it together into one tensor
        return tf.stack(_resulting_list), fw_labels

    def _bert_sentence_decoder(self, forward_labels: tf.Tensor, layer_name: str = 'bert_sentence_decoder') -> tf.Tensor:
        raise NotImplementedError()

    def _positional_encode_sequences(self, train_inputs: tf.Tensor) -> tf.Tensor:
        """

        Parameters
        ----------
        train_inputs: tf.Tensor

        Returns
        -------
        tf.Tensor
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
            ## YOU SHOULD SAY SOME WARNING THAT YOU DO NOT HAVE either cuda OR _time_major

            # NOTE:
            # - the result of the `tf.contrib.cudnn_rnn.CudnnGRU` is a touple
            # - first part of that tuple is the tensor of the shape:
            # IF TIME  MAJORIZED: (sequence_length, batch_size, 2*_latent_space_dimension), if you use biderictional
            # OR (sequence_length, batch_size, _latent_space_dimension) if you use unidirectional
            #
            # - the second part of that tuple is THE TUPLE of the type ('final output',)
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
            if self._time_major:
                _encoder_out = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=self._number_rnn_in_layers,
                                                             num_units=self._latent_space_dimension,
                                                             direction=self._direction)(_encoder_in)[0]
            else:
                raise ValueError('`cuda` expects time majorozation!')
        else:
            # ok I am doing just unidirectional, so I need only th forward cell
            _fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._latent_space_dimension)

            _sequence_length = tf.reduce_sum(tf.sign(train_inputs), reduction_indices=int(not self._time_major))

            if self._direction == 'bidirectional':
                _bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self._latent_space_dimension)

                _encoder_out = tf.nn.bidirectional_dynamic_rnn(_fw_cell, _bw_cell, _encoder_in,
                                                               sequence_length=_sequence_length, dtype=tf.float32,
                                                               time_major=self._time_major)

                # you must put those two tensor at the top of each other; so you will be consistent with CUDA NN
                _encoder_out = tf.concat(_encoder_out[0], axis=-1)

            elif self._direction == 'unidirectional':
                _encoder_out = tf.nn.dynamic_rnn(_fw_cell, _encoder_in, sequence_length=_sequence_length,
                                                 dtype=tf.float32, time_major=self._time_major)

                _encoder_out = _encoder_out[0]

            else:
                raise NotImplementedError('For RNN direction, you can use either `unidirectional` or `bidirectional`.')
            # TO DO ! ???? i am not sure whether this is still true
            # here synchronize the api of the else branch with the api that you got from
            # the cuda branch, because current _encoder_out !=api=! _encoder_out (in cuda)

        return _encoder_out

    def _matmul_ready_encoded_batch_tensor(self, tensor) -> tf.Tensor:
        """

        Parameters
        ----------
        tensor
            is the tf encoded tensor, it can have input shapes: (time_dimension, batch_size, embedding_dimension)
            if self._time_major or (ADD DIMENSIONS)

        Returns
        -------
        tensor
            returns the batch tf.matmul -  ready tensor; i.e. will have dimension
            (batch_size, time_dimension, embedding_dimension)

        """
        if self._time_major:
            # this does the following transformation
            # input is : (time_dimension, batch_size, embedding_dimension)
            # output will be : (batch_size, time_dimension, embedding_dimension)
            return tf.transpose(tensor, perm=[1, 0, 2])

        else:
            raise NotImplementedError

    def _attention(self, name: str, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
                   dropout: bool) -> (tf.Tensor, tf.Tensor):
        """

        Parameters
        ----------
        name: str
            is the name of this attention layer
            this layer's unormalized probabilities are stored in the instance memmory
        query

        key: tf.Tensor

        value: tf.Tensor

        dropout: bool

        Returns
        -------
        tuple
            is a tuple; where the first entry is the calculated scaled attention:
            Attention(Q, K, V) ~ Softmax((Q * K^T)/sqrt(d_k)) * V

        """
        # -- step 1 -- dimension d_k

        # this is getting the time dimension
        # _d_k = query.get_shape()[1]
        _d_k = tf.to_float(tf.shape(query)[-1])

        # -- step 2 -- calculate scores
        _scores = (query @ tf.transpose(key, perm=[0, 2, 1])) / tf.math.sqrt(_d_k)

        # calculate attention unorm. probability
        # and upload it to the instance memory
        _att_u_probability = tf.nn.softmax(_scores, axis=-1, name=name)

        # is this upload correct and doing any good?
        # Y, at this point I do not know how to get the intermediate layer, maybe by eval??
        # maybe ONLY if you are using the eager evaluation??

        #self._attention_layers[name] += [tf.identity(_att_u_probability, name = name)]
        if dropout is not None:
            _att_u_probability = tf.nn.dropout(_att_u_probability, keep_prob=self._dropout_prob)

        return _att_u_probability @ value

    def _multi_head_attention(self, name_of_attention: str, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
                              num_heads: int, batch_size: int, d_model: int,
                              dropout: bool = None) -> (tf.Tensor, tf.Tensor):
        """
        Parameters
        ----------
        name_of_attention:str
            is the name of this attention layer

        mask

        dropout


        note, we want to have: d_model % num_heads == 0
        in this implementation we are setting: d_q == d_k == d_v == d_model/num_heads

        """
        # -- fun-globals --
        _attention_layer_name = 'encoder_attentions'

        # -- step 0 -- calculate the effective `head dimension`

        # also define the `d_q`, `d_k` and `d_v`  == d_model/num_heads
        _d_q = _d_k = _d_v = int(d_model / num_heads)

        # -- step 1-- define matrix `W_i_Q` and `W_i_K` and W_i_V
        # it will have a shape (num_heads, d_model)
        #
        # ??? maybe the way how those tensors are initialized are not good??
        W_i_Q = tf.get_variable("W_i_Q_{}".format(name_of_attention), shape=[num_heads, batch_size, d_model, _d_q],
                                initializer=tf.initializers.glorot_normal(),
                                trainable=True)

        W_i_Q = tf.unstack(W_i_Q, axis=0)

        W_i_K = tf.get_variable('W_i_K_{}'.format(name_of_attention), shape=[num_heads, batch_size, d_model, _d_k],
                                initializer=tf.initializers.glorot_normal(),
                                trainable=True)
        W_i_K = tf.unstack(W_i_K, axis=0)

        # print('~~~> ',W_i_K[0].shape)
        W_i_V = tf.get_variable('W_i_V_{}'.format(name_of_attention), shape=[num_heads, batch_size, d_model, _d_v],
                                initializer=tf.initializers.glorot_normal(),
                                trainable=True)

        W_i_V = tf.unstack(W_i_V, axis=0)

        # this matrix is basically an overall metric that acts on the concatenation of all attentions from all heads
        W_O = tf.get_variable('W_O_{}'.format(name_of_attention), shape=[batch_size, d_model, d_model],
                              initializer=tf.initializers.glorot_normal(), trainable=True)
        # print(W_O.shape)
        # print('--len-> ', list(zip(W_i_Q, W_i_K, W_i_V)))

        # -- step 2 -- calculate the generalized attention
        # here you are getting tuples of an attention as well as the self attentions

        # you must transpose query, key, and value, because the last dimension is time and not a _d_model

        # TO DO::: check what are the best params of this function
        #
        # NOTE: very important:
        # - if you are using `map_fn`, then if you don't state explicit dtype return structure
        # it is assumed that the return structure is the same as input structure
        # if you want to return different data structurel; you must specify it explicitly! As I did.
        # NOTE: the result of this map is a tuple of tensors
        # _X = tf.stack(list(zip(W_i_Q, W_i_K, W_i_V)))

        # NOTE we must STACK THEM AS ABOVE; since the tf.map_fn ONLY UNSTACK BY THE 0-th DIMENSION
        # !!! also the mask here is problematic, since in the function self._attention I do not have parameter mask
        _attentions = tf.map_fn(lambda W: self._attention(_attention_layer_name, query @ W[0], key @ W[1], value @ W[2],
                                                          dropout), tf.stack(list(zip(W_i_Q, W_i_K, W_i_V))),
                                dtype=tf.float32)

        # query 33, 42, 512
        # W[0] 33, 512, 64
        # 33, 42, 64
        # print('---> ', (query@W_i_Q[0]).shape)
        # print('---> ', _attention(query@W_i_Q[0], key@W_i_K[0], value@W_i_V[0], mask, dropout)[0].shape)

        # ??? i am not sure whether this is the best appropach ???
        # _self_attentions = list(map(lambda x: x[-1], _attentions_X))
        # _attentions =  list(map(lambda x: x[0], _attentions_X))

        # print('---> ', _attentions.shape)

        # this should have dimension ()
        _multi_h_attention = tf.concat(tf.unstack(_attentions, axis=0), axis=-1)

        # print('---> ', _multi_h_attention.shape)

        # now the concatenated tonsors should have dimensions
        # (_batch_size, _d_time, _d_model)

        # -- step 3 -- multiply by the output matrix
        return _multi_h_attention @ W_O

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
