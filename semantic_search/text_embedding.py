import re, string
from typing import List, Dict
import itertools
from collections import Counter
import numpy as np

# loading glove
from cogworks_data.language import get_data_path
from gensim.models import KeyedVectors
filename = "glove.6B.200d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

#identifies all punction character
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.
        
    Parameters
    ----------
    corpus : str

    Returns
    -------
    str
        the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def to_token(caption: str) -> List[str]:
    """ Makes captions/queries into tokens by word.

    Process captions/queries by lowercasing the text, removing punctuation, 
    and tokenizing words based on white space.

    Parameters
    ----------
    caption: str

    Returns
    -------
    list
        the caption is split into individual words"""
    return sorted(set(strip_punc(caption.lower()).split()))

def make_the_vocab(token_ls: List[List[str]]) -> List[str]:
    """ Primary code block that enables the making of a vocab list of all words in COCO dataset

    Parameters
    ----------
    ls: List[List[str]]
        Each inner list holds the tokens of one caption from the COCO dataset
        Outer list encompasses all the inner lists

    Returns
    -------
    list
        all the words are compiled into one list"""
    return sorted(set(itertools.chain(*token_ls)))

def to_idf(token_ls: List[List[str]]) -> List[float]:
    N = len(token_ls)
    # print(N)
    # counts number of documents a word appears in
    flat_token = sorted(itertools.chain(*token_ls))
    nt_counter = dict(sorted(Counter(flat_token).items()))
    # gets alphabetically sored num occurances
    nt = np.array(list(nt_counter.values()), dtype=float)
    idf = np.log10(N / nt)
    return idf

def make_idf_mapping(idf: List[float], vocab:List[str]) -> Dict[str, float]:
    return {v:i for v, i in zip(vocab, idf)}

def get_glove(word: str):
    """Returns word's GloVe embedding, or a zero vector if word is OOV for GloVe."""
    if word in glove:
        return glove[word]
    return np.zeros(glove.vector_size)

def make_caption_descriptor(caption: str, idf_map: Dict[str, float]):
    d = np.zeros(glove.vector_size)
    for t in to_token(caption):
        if t in idf_map:
            d += idf_map[t] * get_glove(t)
    # words absent from idf_map/GloVe leave d untouched; guard against
    # normalizing an all-zero vector (caption with no known words)
    norm = np.linalg.norm(d)
    if norm > 0:
        d /= norm
    return d