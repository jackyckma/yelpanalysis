#!/bin/bash
sudo docker exec yelp-dev /yelpanalysis/script/query.py $1 "$2"