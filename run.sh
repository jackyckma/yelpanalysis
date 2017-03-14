#!/bin/bash

if [ "$1" == "-b" ];then
  echo "Build Image: yelpanalysis"
  docker build -t yelpanalysis .
fi

docker images
docker rm -f yelp-dev
docker run -td -p 8888:8888 -p 27017:27017 --name yelp-dev -v `pwd`/data:/datamount yelpanalysis mongod --bind_ip=0.0.0.0

if [ -f "data/yelp_academic_dataset_business.json" ] &&
   [ -f "data/yelp_academic_dataset_checkin.json" ] &&
   [ -f "data/yelp_academic_dataset_review.json" ] &&
   [ -f "data/yelp_academic_dataset_tip.json" ] &&
   [ -f "data/yelp_academic_dataset_user.json" ]; then
   echo "json files exist. untar skipped."

   attempt=0
   while [ $attempt -le 59 ]; do
      attempt=$(( $attempt + 1 ))
      echo "Waiting for server to be up (attempt: $attempt)..."
      result=$(docker logs yelp-dev)
      if grep -q 'waiting for connections on port 27017' <<< $result ; then
         echo "Mongodb is up!"
         break
      fi
   sleep 5
done

else
   echo "untar compressed file."
   docker exec yelp-dev tar xvzf /datamount/yelp_dataset_challenge_round9.tar -C /datamount
fi

docker exec yelp-dev mongoimport --db yelp --collection business /datamount/yelp_academic_dataset_business.json
docker exec yelp-dev mongoimport --db yelp --collection checkin /datamount/yelp_academic_dataset_checkin.json
docker exec yelp-dev mongoimport --db yelp --collection review /datamount/yelp_academic_dataset_review.json
docker exec yelp-dev mongoimport --db yelp --collection tip /datamount/yelp_academic_dataset_tip.json
docker exec yelp-dev mongoimport --db yelp --collection user /datamount/yelp_academic_dataset_user.json

docker exec yelp-dev mkdir /yelpanalysis
docker cp script yelp-dev:/yelpanalysis/
docker exec yelp-dev python /yelpanalysis/script/yelpanalysis.py
#docker exec -i -t yelp-dev /bin/bash

