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
            {'$sample':{'size':2000}},
            {'$project':{'id':1, 'business_id':1, 'name':1, 'address':1, 'stars':1, 'review_count':1, 'categories':1}},
            ]
        l_business = list(self.db.business.aggregate(pipeline))
        
        l_b_id = [record['business_id'] for record in l_business]

        # Reviews
        pipeline = [
            {'$match':{'business_id':{'$in':l_b_id}}},
            {'$project':{'id':1, 'business_id':1, 'user_id':1, 'useful':1, 'stars':1, 'text':1}}
            ]
        record_set = self.db.review.aggregate(pipeline)
        b_review = collections.namedtuple('b_review', 'business_id user_id useful rev_stars text')
        l_texts = [b_review(business_id=record['business_id'],
                          user_id=record['user_id'],
                          useful=record['useful'],
                          rev_stars=record['stars'],
                          text=record['text']) for record in record_set]
        return l_business, l_texts

    
class YelpModel(object):
    def __init__(self, l_text, df_context):
        self.df_context = df_context
        self.l_text = l_text
        
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

        tokenizer = Tokenizer()
        return tokenizer.tokenize(l_text)
    
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
    def __init__(self, l_text, df_context):
        self.l_text = l_text
        self.df_context = df_context
    
    def train(self):
        model = models.doc2vec.Doc2Vec(size=150, min_count=2, iter=55)
        model.build_vocab(self.process_input(self.l_text, self.df_context))
        model.train(self.process_input(self.l_text, self.df_context))
        self.model = model
        return self.model
        
    def process_input(self, l_text, df_context):
        for idx, item in enumerate(l_text):
            record = df_context.iloc[idx]
            tags = record['categories'] + \
                                  [record['name'],
                                   record['stars'],
                                   record['rev_stars']]  
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
    df_context = pd.DataFrame([{'business_id':record.business_id,
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

    print('demo...')
    yelp_doc_model.infer_from_sentence('best burger with good service')
    yelp_doc_model.infer_from_sentence('best burger with cheap price')
    yelp_doc_model.infer_from_sentence('best sushi')
#    yelp_model = YelpModel(l_reviews, df_business)

#    print('top similar in lsi')
#    yelp_model.top_similar('burger is the best!')
#
#    print('w2v')
#    yelp_model.train_w2v()
#    yelp_model.m_w2v.most_similar(positive=['burger', 'cereal'], negative=['lady'], topn=1)
#    yelp_model.m_w2v.doesnt_match("man cereal rib lunch".split())
#    yelp_model.m_w2v.similarity('woman', 'man')
#    yelp_model.m_w2v.similar_by_word('burger', topn=10, restrict_vocab=None)


#    yelp_doc_model = YelpDocumentModel(df_business, yelp_model)
    
#    bv = yelp_doc_model.biz_vector()

    
matches = dv.most_similar('burger')
matches = list(filter(lambda x: 'SENT_' in x[0], matches))

