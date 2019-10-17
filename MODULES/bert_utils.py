"""
Created on Sat Jun 29 2019

@author: polny
"""
import tensorflow as tf
import os
import numpy as np


def parse_and_pad(seq,
                  max_sequence_length: int,
                  eos_token: bool = True) -> tf.Tensor:
    """

    """
    sequence_features = {'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    _, sequence_parsed = tf.parse_single_sequence_example(serialized=seq,
                                                          sequence_features=sequence_features)

    t = sequence_parsed['tokens']

    if eos_token:
        t = tf.pad(t, [[0, 1]], constant_values=3)

    return tf.pad(t, [[0, max_sequence_length - tf.shape(t)[0]]])


def tf_record_iterator(filename: str,
                       max_sequence_length: int = 40,
                       pad: bool = True,
                       eos_token: bool = True,
                       **kwargs) -> 'iter':
    """
    this produces ONE record only; it does not produce batches
    first and foremost;

    since the max_sequence_length:: DOES NOT include the `eos` and `bos` tokens
    you MUST add 2

    Parameters
    ----------

    """
    # you must add 2; because of <eos> and <bos> tokens
    max_sequence_length += 2

    # formaly read the `dataset`
    dataset = tf.data.TFRecordDataset(filename)

    # only if you want to pad sequence; use max_sequence_length and eos token.
    if pad:
        dataset = dataset.map(lambda t, sent_length=max_sequence_length,
                              eos_token=eos_token:
                              parse_and_pad(t, sent_length, eos_token))

    # ?? I do not understand what `prefetch` does ??
    return dataset.prefetch(1).make_one_shot_iterator()


def tf_record_batch_iterator(filename: str,
                             batch_size: int = 48,
                             sub_slice_dimension: int = 2,
                             max_sequence_length: int = 40,
                             pad: bool = True,
                             eos_token: bool = True,
                             time_major: bool = False,
                             **kwargs):
    """
    This function creates an iterator over tf records; but produces a batches
    instead of single examples. It implements the sliding window batch.

    Parameters
    ----------
    filename: str

    max_sequence_length: int

    pad: bool

    eos_token: str

    batch_size: int

    sub_slice_dimension: int
        is the subdivision within the batch windows,
        each iterator provides the a windows (the skipping windows in this function)
        within each of that skipping windows we have a smaller subdivision

    **kwargs


    Returns
    -------
    iterator

    Notes
    -----

    Examples
    --------

    """
    # you must add 2; because of <eos> and <bos> tokens
    max_sequence_length += 2

    # formaly read the `dataset`
    dataset = tf.data.TFRecordDataset(filename)

    # map `parse_and_pad` if requested
    # only if you want to pad sequence; use max_sequence_length and eos token.
    if pad:
        dataset = dataset.map(lambda t, sent_length=max_sequence_length,
                              eos_token=eos_token:
                              parse_and_pad(t, sent_length, eos_token))

    # now, apply the sliding window batch
    # old way:
    #dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=batch_size))
    #dataset = dataset.window(size=batch_size, shift=None, stride=1).flat_map(
    #    lambda x: x.batch(batch_size))
    #dataset = dataset.window(size=batch_size, shift=None, stride=1)

    # now shift it by 3 ??
    # THIS IS VERY IMPORTANT :: THIS BATCHES THE DATA TO A BATCH WITH THE SUBTENSORS OF SIZE 3
    # now we are doing 2, because in bert model we have just current sentence
    # and next sentence we want to translate to?

    #dataset = dataset.batch(batch_size).map(lambda x: x[:2])


    # new way II:: should be better because is not chopping stuff
    # if shift = None - that means that shift = batch_size - that means that from [1,2,3,4,5,6] we have [1,2,3], [4,5,6]
    # stride 1 means no holes in a windows
    #
    # more about above `stride` and `shift` parameters here:
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/sliding_window_batch
    dataset = dataset.window(size=batch_size, shift=None, stride=1,\
                            drop_remainder=True).flat_map(lambda x: x.batch(sub_slice_dimension))

    dataset = dataset.batch(batch_size)

    if time_major:
        # before we had ??
        # maybe before we had
        # NOTE:: if you are doing line
        # `dataset = dataset.batch(batch_size).map(lambda x: x[:3])`
        # THEN you must do the following
        #dataset = dataset.map(lambda x: tf.transpose(x, perm=[0, 2, 1]))

        dataset = dataset.map(lambda x: tf.transpose(x, perm=[1, 2, 0]))

        # if you are not doing the line
        # `dataset = dataset.batch(batch_size).map(lambda x: x[:3])`
        # then you uncomment the following line
        #dataset = dataset.map(lambda x: tf.transpose(x, perm=[1, 0]))
    else:
        dataset = dataset.map(lambda x: tf.transpose(x, perm=[1, 0, 2] ))
        #dataset = dataset.map(lambda x: tf.transpose(x, perm=[2, 0, 1]))

    # prefetching means you are moving this data into the ram, or another fast
    # memory. We are prefatching by 1; just because we have 1 batch that is gonna
    # be eaten by the training algorithm.
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator()


def inspect_batch_of_tf_record_files(path_to_tf_records: str,
                                     name_of_tf_records: str,
                                     batch_size: int = 142,
                                     **kwargs):
    """
    Parameters
    ----------
    path_to_tf_records: str

    name_of_tf_records: str


    Raises
    ------

    """
    with tf.Session() as sess:

        # create an iterator over the dumped files
        tf_record_iter = tf_record_batch_iterator(os.path.join(path_to_tf_records,
                                                               name_of_tf_records),
                                                  batch_size=batch_size,
                                                  **kwargs)

        # read and print the tf_records
        try:
            # if you want you can print whatever characteristics here.
            print(sess.run(tf_record_iter.get_next()))
        except tf.errors.OutOfRangeError:
            raise


def inspect_tf_record_files_one_by_one(path_to_tf_records: str,
                                       name_of_tf_records: str,
                                       upper_bound: int = 42,
                                       **kwargs):
    """

    Parameters
    ----------
    path_to_tf_records: str =
    name_of_tf_records: str =

    upper_bound: int
        how many records should be printed, default is 42.

    **kwargs = are optional named parameters, that hold the params one wants to
    pass into `tf_record_iterator`.

    Notes
    -----
    - this function is slow; probably its better to use the
    `inspect_batch_of_tf_record_files` instead of this one.
    - this function implicitly assumes that the tf record that was dumped
    was dumped using function `wikipedia_corpus.produce_tf_sequence_w2v` or
    similar.
    """
    with tf.Session() as sess:

        # create an iterator over the dumped files
        tf_record_iter = tf_record_batch_iterator(os.path.join(path_to_tf_records, name_of_tf_records), **kwargs)

        for ind in range(upper_bound):
            try:
                print(sess.run(tf_record_iter.get_next()))
            except tf.errors.OutOfRangeError:
                break


def apply_random_fill_mask(in_tensor: tf.Tensor, in_tensor_shape: tuple, replacement:(int, float),
                           replacement_rate: float = 0.2) -> tf.Tensor:
    """

    Parameters
    ----------
    in_tensor: tf.Tensor
        is the TensorFlow Variable to be changed using the random boolean mask,
        by this tensor we mean int or float type of tensor.

    in_tensor_shape: tuple
        is the tuple representing the shape of the in_tensor

    replacement: int or float
    is the the mask token to be used to actually replace parts of the tensor,
    it is of the same type as members of `in_tensor`.

    replacement_rate: float
        is the fraction of elements to be replaced (must be between 0 (inclusive) and 1 (inclusive)),
        -- default -- is 0.2

    Returns
    -------
    tf.Tensor
        is the TensorFlow variable with replaced members using random boolean mask

    Examples
    --------

    """
    assert (1.0 >= replacement_rate >= 0.0), f'The replacement rate you provided: {replacement_rate}' \
        f'does not represent probability.'

    # -- step  0 -- create boolean mask
    _bool_mask = np.random.choice([0, 1], size=in_tensor_shape,
                                  p=((1. - replacement_rate), replacement_rate)).astype(np.bool)

    _bool_mask = tf.constant(~_bool_mask, dtype=tf.bool)

    # -- get the full masking tensor --
    _masking_tensor = replacement*tf.ones_like(in_tensor)

    return tf.where_v2(_masking_tensor, in_tensor, _bool_mask)
