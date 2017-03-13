# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from pymongo import MongoClient
import logging
from gensim import corpora, models, similarities
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import time
import numpy as np

"""
References:

Item2vec: neural item embedding for collaborative filtering
https://arxiv.org/pdf/1603.04259.pdf

From word embeddings to item recommendation
https://arxiv.org/pdf/1601.01356.pdf

"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
last_time = time.time()

def elapsed_time():
    global last_time
    new_time = time.time()
    time_diff = new_time - last_time
    last_time = new_time
    return time_diff
    
class YelpData(object):
    def __init__(self, city, ip_address, port=27017):
         
        self.db = MongoClient(ip_address, port).yelp
        self.city = city

        
    def business(self):
        # Business ID
        pipeline = [
            {'$match':{'stars':{'$gte':4},
                       'city':self.city,
                       'categories':'Restaurants'}},
            {'$project':{'id':1, 'business_id':1, 'name':1, 'address':1, 'stars':1, 'review_count':1, 'categories':1}},
            ]
        record_set = self.db.business.aggregate(pipeline)
        df = pd.DataFrame(list(record_set))
        self.df_business = df
        return df

    def review(self, tokenize=True):
        # Reviews
        l_business = list(self.business()['business_id'].values)
        pipeline = [
            {'$match':{'business_id':{'$in':l_business}}},
            {'$project':{'id':1, 'business_id':1, 'text':1}}
            ]
        record_set = self.db.review.aggregate(pipeline)
        b_review = collections.namedtuple('b_review', 'business_id text')
        texts = [b_review(business_id=record['business_id'], 
                          text=record['text']) for record in record_set]

        return texts
    
class YelpModel(object):
    def __init__(self, l_business_reviews):
        self.l_business_reviews = l_business_reviews
        self.l_business = [record.business_id for record in l_business_reviews]
        self.l_text = [record.text for record in l_business_reviews]
        
        print('tokenizing...')
        self.l_tokenized_text = self.tokenize(self.l_text)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
        
        self.m_dict = None
        self.l_bow = None
        self.m_tfidf = None
        self.l_tfidf = None
        self.m_lsi = None
        self.l_lsi = None
        self.m_w2v = None
        
    def train_dictionary(self):
        print('building dictionary...')
        self.m_dict = corpora.dictionary.Dictionary()
        self.m_dict.add_documents(self.l_tokenized_text)   
        self.l_bow = self.get_bow(self.l_tokenized_text)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
        
    def train_tfidf(self):
        print('calculating tfidf...')
        if self.l_bow is None:
            self.train_dictionary()
            
        self.m_tfidf = models.TfidfModel(self.l_bow)
        self.l_tfidf = self.get_tfidf(self.l_bow)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
        
    def train_lsi(self):
        print('building lsi model...')
        if self.l_tfidf is None:
            self.train_tfidf()
        if self.m_dict is None:
            self.train_dictionary()
            
        self.m_lsi = models.LsiModel(self.l_tfidf, id2word=self.m_dict, num_topics=200)
        self.l_lsi = self.get_lsi(self.l_tfidf)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
        
        print('building lsi simiarity matrix...')
        self.m_sim = similarities.MatrixSimilarity(self.l_lsi)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
        
    def train_w2v(self):
        # word2vec
        print('training word2vec model...')
        self.m_w2v = models.word2vec.Word2Vec(self.l_tokenized_text, size=250, workers=4)
        print('elapsed time:[{:.2f}s]'.format(elapsed_time()))

    
    def tokenize(self, l_text):

        tokenizer = RegexpTokenizer(r'\w+')
        stoplist = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()
        
        def _tokenize(in_text):
            tokens = tokenizer.tokenize(in_text.lower())
            stopped_tokens = [i for i in tokens if not i in stoplist]
#            return [p_stemmer.stem(i) for i in stopped_tokens]
            return stopped_tokens

        l_tokenized_text = [_tokenize(text) for text in l_text]
        return l_tokenized_text

    
    def get_bow(self, l_tokenized_text):
        return [self.m_dict.doc2bow(tokenized_text) for tokenized_text in l_tokenized_text] 
    
    def get_tfidf(self, l_bow):
        return self.m_tfidf[l_bow]
    
    def get_lsi(self, l_tfidf):
        return self.m_lsi[l_tfidf]

    
    def lsi_similarity(self, doc):
#        doc = "Best Ribeye steak! Friendly and good environment!"
        
        vec_bow = self.get_bow(doc.lower().split())
        vec_tfidf = self.get_tfidf(vec_bow)
        vec_lsi = self.get_lsi(vec_tfidf)

        sims = self.m_sim[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        return sims

    def top_similar(self, doc, n=10):
        sims = self.lsi_similarity(doc)
        
        for i in range(n):
            print(self.l_text[sims[i][0]])
            print('')
            
            
class YelpDocumentModel(object):
    def __init__(self, df_business, yelp_model):
        self.df_business = df_business
        self.yelp_model = yelp_model

    def doc_vector(self, words):
        m_wv = self.yelp_model.m_w2v
        word_collection = set(m_wv.wv.index2word)
        l = [m_wv[word] for word in words if word in word_collection]
        return np.mean(l, axis=0)
           
    def biz_vector(self):
        l = yelp_model.l_business
        d={}
        for idx, item in enumerate(l):
            d[item] = d.get(item, []) + [idx]

        d2={}
        
        _i = 0
        for key in d:
            l_reviews = [self.yelp_model.l_tokenized_text[i] for i in d[key]]
            l_reviewvecs = [self.doc_vector(review) for review in l_reviews]
            d2[key] = np.mean(l_reviewvecs, axis=0)
            _i += 1
            print('{0}:{1}'.format(_i, key))

        return d2
        
if __name__=='__main__':

    print('loading data...')
    yelp_data = YelpData(ip_address='localhost', city='Las Vegas')
    df_business = yelp_data.business()
    l_reviews = yelp_data.review()
    print('elapsed time:[{:.2f}s]'.format(elapsed_time()))

    print('building models...')
    yelp_model = YelpModel(l_reviews)

#    print('top similar in lsi')
#    yelp_model.top_similar('burger is the best!')
#
    print('w2v')
    yelp_model.train_w2v()
    yelp_model.m_w2v.most_similar(positive=['burger', 'cereal'], negative=['lady'], topn=1)
    yelp_model.m_w2v.doesnt_match("man cereal rib lunch".split())
    yelp_model.m_w2v.similarity('woman', 'man')
    yelp_model.m_w2v.similar_by_word('burger', topn=10, restrict_vocab=None)


    yelp_doc_model = YelpDocumentModel(df_business, yelp_model)
    
    bv = yelp_doc_model.biz_vector()

    


