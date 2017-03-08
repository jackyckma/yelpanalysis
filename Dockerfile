FROM continuumio/anaconda
MAINTAINER Jacky MA <jacky.ckma@gmail.com>
RUN apt-get -qq update
RUN git clone https://github.com/jackyckma/yelpanalysis.git /yelpanalysis
ADD data/yelp_dataset_challenge_round9.tar /yelpanalysis/data
CMD echo Docker image for Yelp Data Analysis
