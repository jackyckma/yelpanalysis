FROM continuumio/anaconda
MAINTAINER Jacky MA <jacky.ckma@gmail.com>
RUN apt-get -qq update
RUN pip install --upgrade gensim pymongo

# Install MongoDB.
RUN \
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10 && \
  echo 'deb http://downloads-distro.mongodb.org/repo/ubuntu-upstart dist 10gen' > /etc/apt/sources.list.d/mongodb.list && \
  apt-get install -y mongodb && \
  mkdir -p /data/db


# Expose ports.
#   - 27017: process
#   - 28017: http
EXPOSE 27017
EXPOSE 28017

VOLUME ["/datamount"]

RUN git clone https://github.com/jackyckma/yelpanalysis.git /yelpanalysis



