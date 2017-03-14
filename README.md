# Yelp Data Analysis

## DESCRIPTION
The program trys to build a recommender system/query system based on users comments. Doc2Vec is employed
on the reviews to build a docvec model. Querys can be conducted either with label (i.e. name of a restuarant)
or with sentence (e.g. 'best burger restaurant').


## STRUCTURE
All the scripts are in the GitHub repositary. However, due to license concern, user have to download the
data from Yelp in order to run the program.

When running the script (run.sh), a docker image will be built and spin up. Libraries, supporting facilities
(mongodb), python scripts and data will be installed or copied into the container. The training of 
Doc2vec model will be invoked and saved in /tmp/

User can then submit queries to the container through docker interface:
sudo docker exec yelp-dev python /yelpanalysis/script/query.py l 'Burger King'
sudo docker exec yelp-dev python /yelpanalysis/script/query.py s 'Best burger place in the town'


## EVALUATION
While doc2vec is a new technology trying to capture the sementics of paragraphs, in our case, the result
doesn't seem really good. Quite often when user searching for a particular food (e.g. burger) and there
will be other kind of restaurants in the results. One of the reason is that the model trys to capture 
similarities between 'reviews' but food type is just one of the many dimensions such as service, speed,
price, location, ambient, etc. Moreover, the author has not yet tune the modelling parameters in detail,
which might results in suboptimal predictions as well.

Nonetheless, sementic modelling of paragraph text should have a wide range of application for the retail
industry: sentiment prediction, social-platform analysis, trend prediction from bloggers, etc. The author
believes its a technique worth drilling into.


## SETUP
1. Git clone the project to local harddisk
git clone https://github.com/jackyckma/yelpanalysis.git

2. Download the Yelp data set
Get the tar file (yelp_dataset_challenge_round9.tar) from https://www.yelp.com/dataset_challenge/dataset and put it in the project subfolder 'data/'

3. BUILD and PREPARE docker image: 
sudo ./run.sh -b
Note: the training of doc2vec model is time consuming, it takes about 10 minutes on AWS m4.xlarge instance.

4. SUBMIT query to docker image: 
sudo docker exec yelp-dev python /yelpanalysis/script/query.py l 'Burger King'
sudo docker exec yelp-dev python /yelpanalysis/script/query.py s 'Best burger place in the town'
