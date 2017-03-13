# -*- coding: utf-8 -*-

import pandas as pd
from pymongo import MongoClient
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



    
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
        df = pd.DataFrame(record_set)
        return df

    def review(self, tokenize=True):
        # Reviews
        l_business = set(self.business()['business_id'].values)
        pipeline = [
            {'$match':{'business_id':{'$in':l_business}}},
            {'$project':{'id':1, 'business_id':1, 'text':1}}
            ]
        record_set = self.db.review.aggregate(pipeline)
        texts = [(record['business_id'], record['text']) for record in record_set]

        return texts
    
class YelpModel(object):
    def __init__(self, l_business_reviews):
        self.l_business_reviews = l_business_reviews
        self.l_business = [record['business_id'] for record in l_business_reviews]
        self.l_text = [record['text'] for record in l_business_reviews]
        
        self.l_tokenized_text = self.tokenize(self.l_text)
        
        self.corp_dict = corpora.dictionary.Dictionary()
        self.corp_dict.add_documents(self.l_tokenized_text)   

        self.l_bow = self.get_bow(self.l_tokenized_text)
        self.corpus_tfidf = self.get_tfidf(self.l_bow)

    
    def tokenize(self, l_text):

        tokenizer = RegexpTokenizer(r'\w+')
        stoplist = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()
        
        def _tokenize(in_text):
            tokens = tokenizer.tokenize(in_text.lower())
            stopped_tokens = [i for i in tokens if not i in stoplist]
            return [p_stemmer.stem(i) for i in stopped_tokens]
#            return stopped_tokens

        l_tokenized_text = [_tokenize(text) for text in l_text]
        return l_tokenized_text

    
    def get_bow(self, l_tokenized_text):
        return [self.corp_dict.doc2bow(tokenized_text) for tokenized_text in l_tokenized_text] 
    
    def get_tfidf(self, l_bow):
        tfidf = models.TfidfModel(l_bow)
        corpus_tfidf = tfidf[l_bow]
        return corpus_tfidf
    
    def get_lsi(self, corpus_tfidf):
        lsi = models.LsiModel(corpus_tfidf, id2word=self.corp_dict, num_topics=200)
        corpus_lsi = lsi[corpus_tfidf]
        return corpus_lsi
    
    def lsi_similarity(self, doc):
#        doc = "Best Ribeye steak! Friendly and good environment!"
        index = similarities.MatrixSimilarity(lsi[corpus])

        vec_bow = corp_dict.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow] # convert the query to LSI space
        print(vec_lsi)
        
        
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        for i in range(10):
            print(texts[sims[i][0]])
            print('')


if __name__=='__main__':

    
    yelp_data = YelpData(ip_address='192.168.56.102', city='Las Vegas')
    df_business = yelp_data.business()
    l_reviews = yelp_data.review()






# word2vec

word2vec_model = models.word2vec.Word2Vec(tokenized_texts, size=250, workers=4)

word2vec_model.most_similar(positive=['burger', 'cereal'], negative=['lady'], topn=1)

word2vec_model.doesnt_match("man cereal rib lunch".split())

word2vec_model.similarity('woman', 'man')

word2vec_model.similar_by_word('salami', topn=10, restrict_vocab=None)

#word_vectors = word2vec_model.wv
#del model