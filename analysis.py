# -*- coding: utf-8 -*-


import json
import pandas as pd

file_business = 'data/yelp_academic_dataset_business.json'
file_checkin = 'data/yelp_academic_dataset_checkin.json'
file_review = 'data/yelp_academic_dataset_review.json'
file_tip = 'data/yelp_academic_dataset_tip.json'
file_user = 'data/yelp_academic_dataset_user.json'


with open(file_user) as json_data:
    data = json_data.read()
    
