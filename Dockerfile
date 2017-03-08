FROM continuumio/anaconda
MAINTAINER Jacky MA <jacky.ckma@gmail.com>
RUN apt-get -qq update
RUN git clone https://github.com/jackyckma/yelpanalysis.git /yelpanalysis
RUN pip install --upgrade gensim
ADD data/yelp_dataset_challenge_round9.tar /yelpanalysis/data


# Install MongoDB.
RUN \
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10 && \
  echo 'deb http://downloads-distro.mongodb.org/repo/ubuntu-upstart dist 10gen' > /etc/apt/sources.list.d/mongodb.list && \
  apt-get update && \
  apt-get install -y mongodb-org && \
  rm -rf /var/lib/apt/lists/*

# Define mountable directories.
VOLUME ["/data/db"]

# Define working directory.
WORKDIR /data

# Expose ports.
#   - 27017: process
#   - 28017: http
EXPOSE 27017
EXPOSE 28017

RUN mongoimport --db yelp --collection business /yelpanalysis/data/yelp_academic_dataset_business.json
RUN mongoimport --db yelp --collection checkin /yelpanalysis/data/yelp_academic_dataset_checkin.json
RUN mongoimport --db yelp --collection review /yelpanalysis/data/yelp_academic_dataset_review.json
RUN mongoimport --db yelp --collection tip /yelpanalysis/data/yelp_academic_dataset_tip.json
RUN mongoimport --db yelp --collection user /yelpanalysis/data/yelp_academic_dataset_user.json

CMD echo Docker image for Yelp Data Analysis
# Define default command.
# CMD ["mongod"]


