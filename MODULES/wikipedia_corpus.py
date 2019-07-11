"""
Created on Sat Jun 29 2019

@author: polny
"""
import re
# from _pytest.monkeypatch import MonkeyPatch
import numpy as np
import gensim
import os
import tensorflow as tf
import time


def wiki_corpus_iterator(path_to_wiki_zipped_corpus: str) -> 'iter':
    """
    ADD DOCSTRING
    """
    # this is a regex that will monkeypatch the gensim-own generator
    # point is that we need to preserve `.`; that is otherwise
    # disregarded
    PAT_ALPHABETIC = re.compile(r'(((?![\d])([.]{,1})[\w|.])+)', re.UNICODE)

    # create the monkeypatch
    ###
    # monkeypatch = _pytest.monkeypatch.MonkeyPatch()
    # monkeypatch.setattr(gensim.utils, 'PAT_ALPHABETIC', PAT_ALPHABETIC)
    gensim.utils.PAT_ALPHABETIC = PAT_ALPHABETIC

    # create 'WikiCorpus' instance
    wiki = gensim.corpora.WikiCorpus(path_to_wiki_zipped_corpus,
                                     lemmatize=False, token_min_len=2)

    # also set that you want the wikipedia metadata
    wiki.metadata = True

    # this returns a numbered iterator of the structure
    # also note; the tokens now should contain the '.'; but noticed that
    # sometimes it might depend on system; check on small chunk
    # before going crazy.
    # global_article_index, (tokens, (article_native_id, article_title))
    return enumerate(wiki.get_texts())


def produce_tf_sequence_w2v(sentence: str,
                            gensim_w2v_model: 'loaded gensim w2v model',
                            maximum_sentence_length: int = 40,
                            maximum_vocabulary_size: int = 20000,
                            clean: bool = True,
                            **kwargs) -> 'tf sequence example':
    """
    ADD DOCSTRING

    this function takes NAIVELY just firs 20000 words (not ordered by frequencies)

    # TO DO::: in the future; using quargs; add a dictionary of most used words
    using the underlying corpus

    Parametres
    ----------
    ordered_corpus_tokens: np.ndarray = is None or an ordered list with tokens of words
        related to the gensim_w2v_model.vocab ordered by the underlying corpus
        word frequencies.
    maximum_vocabulary_size: int = is None or integer; determining how many
        of tokens to take. If None; then we will take all tokens. Default is 20000.

    maximum_sentence_length:int = default is 40; is the maximum sentence length; in out case
        we are taking litarally `maximum_sentence_length` but actually number of
        tokens will be `maximum_sentence_length` + 2; because <BOS>= 2 = beginning of sentence
        token and <EOS>= 3 = end of sentence token.

    **kwargs: optional parameters dictionary:
        - clean_sentence_function: callable =
        - ordered_corpus_tokens: np.ndarray =

    this produces tf sequence example of w2v tokens
    """
    # define default cleaning function
    _clean_sentence = kwargs.get('clean_sentence_function', None)

    if _clean_sentence is None:
        def _clean_sentence(sentence): return re.sub(r'[ ]+', ' ',
                                                     re.sub(r'[^A-Za-z0-9]', ' ',
                                                            sentence)).strip()

    # get ordered list of word indices of the gensim model by the usage
    # in the underlying corpus
    _corpus_frequent_token_indices = kwargs.get('ordered_corpus_tokens', None)

    # define bool function that returns true or false depending on whether you
    # have ordered_corpus_tokens np.ndarray or None
    if not(isinstance(maximum_vocabulary_size, int) or maximum_vocabulary_size is None):
        raise ValueError('`maximum_vocabulary_size` must be eithe None or int!')

    # define bool _is_good_index function
    if isinstance(_corpus_frequent_token_indices, np.ndarray):

        # this looks whether you need to constrain the length of the index frequencies
        _indices_subsection = set(_corpus_frequent_token_indices) if \
            maximum_vocabulary_size is None else\
            set(_corpus_frequent_token_indices[:maximum_vocabulary_size])

        def _is_good_index(ind, index_set=_indices_subsection):\
            return ind in index_set

    elif _corpus_frequent_token_indices is None:
        # this is the case when you do not have frequency dictionary in hand
        def _is_good_index(ind): return ind < maximum_vocabulary_size if\
            isinstance(maximum_vocabulary_size, int) else True
    else:
        raise ValueError('`ordered_corpus_tokens` can be only np.ndarray or None!')

    # this will store an unique token for `unknown` word token
    _unknown_word_id = 1

    # clean sentence if necessary
    sentence = _clean_sentence(sentence) if clean else sentence

    # get words in sequence
    _words = sentence.split()

    # initialize tf SequenceExample
    seq = tf.train.SequenceExample()

    # create a feature list tokens
    fl_tokens = seq.feature_lists.feature_list['tokens']

    # we are cutting the number
    # note that this does not ensure the same length sentences;
    # this would need to be solved later in the modeling part
    for word in _words[:maximum_sentence_length]:
        if word in gensim_w2v_model:

            # get word id
            _word_id = gensim_w2v_model.vocab[word].index

            if _is_good_index(_word_id):
                # Note; we add + 4; reason is that we need 4 special tokens
                # <EOS> = 3; <BOS> = 2, <UNK> = 1, <FILL> = 0
                _word_id += 4
            else:
                _word_id = _unknown_word_id
        else:
            _word_id = _unknown_word_id

        # append _word_id to fl_tokens
        fl_tokens.feature.add().int64_list.value.append(_word_id)

    return seq


def dump_wiki_text_as_tf_records(path_to_wiki_zipped_corpus: str,
                                 path_to_gensim_model: str,
                                 path_to_save_tf_records: str,
                                 name_to_save_tf_record: str = 'data.tfrecords',
                                 verbose: bool = True,
                                 use_w2v_orders: bool = False,
                                 **kwargs):
    """
    this function takes the wikipedia corpus and dumps it into the
    tf record format; the tf record format is ready to be ingested by TF.

    Parameters
    ----------
    use_w2v_orders: bool = whether you want to use w2v word frequencies to
        produce word relevances.
    """
    # get from quargs possible values of function `produce_tf_sequence_w2v`
    _maximum_sentence_length = kwargs.get('maximum_sentence_length', 40)
    _maximum_vocabulary_size = kwargs.get('maximum_vocabulary_size', 20000)
    _clean = kwargs.get('clean', True)
    _clean_sentence_function = kwargs.get('clean_sentence_function', None)

    if use_w2v_orders:
        # here you must produce 0-based ordered token indices
        # where they are ordered in descending order by their frequencies wrt
        # the underlying corpus

        # TO DO: !!! implement this !!!
        #_ordered_corpus_tokens = np.array([])

        raise NotImplementedError

    else:
        _ordered_corpus_tokens = None

    # figure out whether the gensim model is binary
    _bin = True if path_to_gensim_model.split('.')[-1] == 'bin' else False

    if verbose:
        print('--> Loading the pre-trained w2v model.')
        _tic = time.time()
    # open the gensim model
    _gensim_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_gensim_model,
                                                                    binary=_bin)

    if verbose:
        print('--> Finished loading the pre-trained w2v model in {} s.'.format(time.time() - _tic))

    if verbose:
        print('--> Starting creation of wikipedia iterator.')
        _tic = time.time()
    # look into the kwargs whether you do not secretly have an iterator
    _wiki_corpus_iter = kwargs.get('wikipedia_corpus_iter', None)

    if _wiki_corpus_iter is None:
        _wiki_corpus_iter = wiki_corpus_iterator(path_to_wiki_zipped_corpus)

    if verbose:
        print('--> Wikipedia iterator prepared in {} s.'.format(time.time() - _tic))

    # also look into kwargs whether you do not have restriction on index
    # this is mainly useful if you have restriction on space// or in debugging session
    _wiki_index_bound = kwargs.get('wiki_index_bound', np.inf)

    # create the file if not yet created
    _saving_path = os.path.join(path_to_save_tf_records, name_to_save_tf_record)

    # create folder if nonexistent
    open(_saving_path, 'a').close()

    # create sent ->tf.sequence mapping function
    def _mapping_function(sentence): return produce_tf_sequence_w2v(sentence,
                                                                    gensim_w2v_model=_gensim_model,
                                                                    maximum_sentence_length=_maximum_sentence_length,
                                                                    maximum_vocabulary_size=_maximum_vocabulary_size,
                                                                    clean=_clean,
                                                                    clean_sentence_function=_clean_sentence_function,
                                                                    ordered_corpus_tokens=_ordered_corpus_tokens)

    #print('WIKI ARTICLE IS : {}'.format(_wiki_corpus_iter[0][1][0]), '\n\n')
    # create the tf record writer
    with tf.python_io.TFRecordWriter(_saving_path) as writer:

        for index, (tokens, (article_id, article_title)) in _wiki_corpus_iter:

            # end looping if index is bigger (or eq) as the _wiki_index_bound
            if index >= _wiki_index_bound:
                break

            # create wikipedia article sentences
            _wiki_article_sentences = ' '.join(tokens).split('.')

            # produce a `seq` object for every sentence
            for seq in map(_mapping_function, _wiki_article_sentences):

                # use writer to write sequences
                writer.write(seq.SerializeToString())
