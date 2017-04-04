# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:33:54 2017

@author: jma


Query interface for doc2vec model
Syntax:
python query.py [l|s] query_str
- [l|s] is the mode of query, 'l' for label query and 's' for sentence query



"""

from gensim import corpora, models, similarities
import argparse

def infer_from_sentence(sentence, n=10):
    """
    Get a sentence input, infer the docvec, and find the similar documents

    Parameters
    ----------
    sentence: string
      sentence for query
    n: int
      number of results to be return

    Returns
    -------
    sims: list of tuples: (doctag, probability)

    """
    inferred_vec = model.infer_vector(sentence.lower().split())
    sims = model.docvecs.most_similar([inferred_vec], topn=n)
    return sims

def infer_from_label(label, n=10):
    """
    Get a label input, infer the docvec, and find the similar documents

    Parameters
    ----------
    label: string
      label for query
    n: int
      number of results to be return

    Returns
    -------
    sims: list of tuples: (doctag, probability)
    """
    vec = model.docvecs[label]
    sims = model.docvecs.most_similar([vec], topn=n)
    return sims


# Load model from /tmp
model = models.doc2vec.Doc2Vec.load('/tmp/d2vmodel.doc2vec')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('mode', help='query mode: [l|s] (label|sentence)')
parser.add_argument('query', help='query string')
args = parser.parse_args()

# Branch to correct function according to the chosen mode
if args.mode in ['l', 'label']:
    rslt = infer_from_label(args.query)
    for t in rslt:
        print(t)
elif args.mode in ['s', 'sentence']:
    rslt=infer_from_sentence(args.query)
    for t in rslt:
        print(t)
else:
    print('Mode can either by [l]abel or [s]entence')