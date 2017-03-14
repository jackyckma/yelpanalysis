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
from sklearn.linear_model import Ridge
import collections
import time
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
last_time = time.time()

def elapsed_time():
    global last_time
    new_time = time.time()
    time_diff = new_time - last_time
    last_time = new_time
    return time_diff

class Tokenizer(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stoplist = set(stopwords.words('english'))
        self.p_stemmer = PorterStemmer()

    def _tokenize(self, in_text):
        tokens = self.tokenizer.tokenize(in_text.lower())
        stopped_tokens = [i for i in tokens if not i in self.stoplist]
#            return [p_stemmer.stem(i) for i in stopped_tokens]
        return stopped_tokens
    
    def tokenize_list(self, l_text):
        l_tokenized_text = [self._tokenize(text) for text in l_text]
        return l_tokenized_text
    
    
class YelpData(object):
    def __init__(self, city, ip_address, port=27017):
         
        self.db = MongoClient(ip_address, port).yelp
        self.city = city

        
    def get_data(self):
        # Business ID
        pipeline = [
            {'$match':{'stars':{'$gte':1},
                       'city':self.city,
                       'categories':'Restaurants'}},
            {'$sample':{'size':3000}},
            {'$project':{'id':1, 'business_id':1, 'name':1, 'address':1, 'stars':1, 'review_count':1, 'categories':1}},
            ]
        l_business = list(self.db.business.aggregate(pipeline))
        
        l_b_id = [record['business_id'] for record in l_business]

        # Reviews
        pipeline = [
            {'$match':{'business_id':{'$in':l_b_id}}},
            {'$project':{'id':1, 'review_id':1, 'business_id':1, 'user_id':1, 'useful':1, 'stars':1, 'text':1}}
            ]
        record_set = self.db.review.aggregate(pipeline)
        b_review = collections.namedtuple('b_review', 'review_id business_id user_id useful rev_stars text')
        l_texts = [b_review(review_id=record['review_id'],
                            business_id=record['business_id'],
                            user_id=record['user_id'],
                            useful=record['useful'],
                            rev_stars=record['stars'],
                            text=record['text']) for record in record_set]
        return l_business, l_texts

           
class YelpDocumentModel(object):
    def __init__(self, l_text, df_context):
        self.l_text = l_text
        self.df_context = df_context
    
    def train(self):
        model = models.doc2vec.Doc2Vec(size=150, min_count=2, iter=10, dm_concat=1, workers=5)
        model.build_vocab(self.process_input(self.l_text, self.df_context))
        model.train(self.process_input(self.l_text, self.df_context))
        self.model = model
        return self.model
        
    def process_input(self, l_text, df_context):
        for idx, item in enumerate(l_text):
            record = df_context.iloc[idx]
            tags = [record['name']]
            yield models.doc2vec.TaggedDocument(l_text[idx], tags)
            
    def infer_from_sentence(self, sentence, n=10):
        inferred_vec = self.model.infer_vector(sentence.lower().split())
        sims = self.model.docvecs.most_similar([inferred_vec], topn=n)
        return sims

    def infer_from_label(self, label, n=10):
        vec = self.model.docvecs[label]
        sims = self.model.docvecs.most_similar([vec], topn=n)
        return sims        


       
        
if __name__=='__main__':

    print('loading data...')
    yelp_data = YelpData(ip_address='localhost', city='Las Vegas')
    l_business, l_reviews = yelp_data.get_data()
    print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
    
    print('tokenizing...')
    tokenizer = Tokenizer()
    l_text = tokenizer.tokenize_list([record.text for record in l_reviews])
    print('elapsed time:[{:.2f}s]'.format(elapsed_time()))
    
    print('merging data...')
    df_context = pd.DataFrame([{'review_id':record.review_id,
                                'business_id':record.business_id,
                                'user_id':record.user_id, 
                                'useful':record.useful,
                                'rev_stars':record.rev_stars} for record in l_reviews])
    df_context = pd.merge(df_context, 
                          pd.DataFrame(l_business),
                          on='business_id')    
    print('elapsed time:[{:.2f}s]'.format(elapsed_time()))


    print('building doc2vec model...')
    yelp_doc_model = YelpDocumentModel(l_text, df_context)
    yelp_doc_model.train()
    print('elapsed time:[{:.2f}s]'.format(elapsed_time()))


    print('saving model...')
    yelp_doc_model.model.save('/tmp/d2vmodel.doc2vec')

