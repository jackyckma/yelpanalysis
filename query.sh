#!/bin/bash
sudo docker exec yelp-dev python /yelpanalysis/script/query.py $1 "$2"
