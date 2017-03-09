# -*- coding: utf-8 -*-

import pandas as pd
from pymongo import MongoClient


client = MongoClient('localhost', 27017)
db = client.yelp

business = db.business
checkin = db.checkin
review = db.review
tip = db.tip
user = db.user


df = pd.DataFrame(list(business.find({}, {'id':1, 'categories':1})))

print df