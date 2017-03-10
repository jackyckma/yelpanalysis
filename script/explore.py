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

client = MongoClient('localhost', 27017)
db = client.yelp

business = db.business
checkin = db.checkin
review = db.review
tip = db.tip
user = db.user

pipeline = [
    {'$project':{'id':1, 'text':1}},
    {'$sample':{'size':50}}    
    ]
k = review.aggregate(pipeline)

texts = review.find({'stars':{'$gt':4}}, {'id':1, 'text':1})

tokenizer = RegexpTokenizer(r'\w+')
stoplist = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

def token_clean(in_text):
    tokens = tokenizer.tokenize(in_text.lower())
    stopped_tokens = [i for i in tokens if not i in stoplist]
    return [p_stemmer.stem(i) for i in stopped_tokens]


corp_dict = corpora.dictionary.Dictionary()


i=0
for text in texts:
    corp_dict.add_documents([token_clean(text['text'])])
    i+=1
    if i%100000==0:
        print(i)

#
#print(corp_dict.token2id)
#
#texts = review.find({}, {'id':1, 'text':1})
#corpus = [corp_dict.doc2bow(text['text']) for text in texts]    
#corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
#
#n_topics=20
#
#ldamodel = models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word = corp_dict, passes=20)
#
#print(ldamodel.print_topics(num_topics=3, num_words=n_topics))