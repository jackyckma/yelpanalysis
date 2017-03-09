#!/bin/bash

if [ "$1" == "-b" ];then
  echo "Build Image: yelpanalysis"
  sudo docker build -t yelpanalysis .
fi

sudo docker images
sudo docker rm -f yelp-dev
sudo docker run -td -p 8888:8888 -p 27017:27017 --name yelp-dev -v `pwd`/data:/datamount yelpanalysis mongod --bind_ip=0.0.0.0

if [ -f "data/yelp_academic_dataset_business.json" ] &&
   [ -f "data/yelp_academic_dataset_checkin.json" ] &&
   [ -f "data/yelp_academic_dataset_review.json" ] &&
   [ -f "data/yelp_academic_dataset_tip.json" ] &&
   [ -f "data/yelp_academic_dataset_user.json" ]; then
   echo "json files exist. untar skipped."
   sleep 20
else
   echo "untar compressed file."
   sudo docker exec yelp-dev tar xvzf /datamount/yelp_dataset_challenge_round9.tar -C /datamount
fi

sudo docker exec yelp-dev mongoimport --db yelp --collection business /datamount/yelp_academic_dataset_business.json
sudo docker exec yelp-dev mongoimport --db yelp --collection checkin /datamount/yelp_academic_dataset_checkin.json
sudo docker exec yelp-dev mongoimport --db yelp --collection review /datamount/yelp_academic_dataset_review.json
sudo docker exec yelp-dev mongoimport --db yelp --collection tip /datamount/yelp_academic_dataset_tip.json
sudo docker exec yelp-dev mongoimport --db yelp --collection user /datamount/yelp_academic_dataset_user.json

sudo docker exec yelp-dev jupyter notebook
sudo docker exec yelp-dev python /yelpanalysis/analysis.py
sudo docker exec -i -t yelp-dev /bin/bash

