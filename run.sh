#!/bin/bash

sudo docker build -t yelpanalysis .
sudo docker images
sudo docker run -td --name yelp-dev -v `pwd`/data:/datamount yelpanalysis mongod

if [ -f "data/yelp_academic_dataset_business.json" ] &&
   [ -f "data/yelp_academic_dataset_checkin.json" ] &&
   [ -f "data/yelp_academic_dataset_review.json" ] &&
   [ -f "data/yelp_academic_dataset_tip.json" ] &&
   [ -f "data/yelp_academic_dataset_user.json" ]; then
   echo "json files exist. untar skipped."
else
   echo "untar compressed file."
   sudo docker exec yelp-dev tar xvzf /datamount/yelp_dataset_challenge_round9.tar -C /datamount
fi

sudo docker exec yelp-dev mongoimport --db yelp --collection business /datamount/yelp_academic_dataset_business.json
sudo docker exec yelp-dev mongoimport --db yelp --collection checkin /datamount/yelp_academic_dataset_checkin.json
sudo docker exec yelp-dev mongoimport --db yelp --collection review /datamount/yelp_academic_dataset_review.json
sudo docker exec yelp-dev mongoimport --db yelp --collection tip /datamount/yelp_academic_dataset_tip.json
sudo docker exec yelp-dev mongoimport --db yelp --collection user /datamount/yelp_academic_dataset_user.json

sudo docker exec python /yelpanalysis/analysis.py
sudo docker exec -i -t yelp-dev /bin/bash

