!/bin/sh
/usr/bin/mongod &&
mongoimport --db yelp --collection business /yelpanalysis/data/yelp_academic_dataset_business.json &&
mongoimport --db yelp --collection checkin /yelpanalysis/data/yelp_academic_dataset_checkin.json &&
mongoimport --db yelp --collection review /yelpanalysis/data/yelp_academic_dataset_review.json &&
mongoimport --db yelp --collection tip /yelpanalysis/data/yelp_academic_dataset_tip.json &&
mongoimport --db yelp --collection user /yelpanalysis/data/yelp_academic_dataset_user.json
