#!/bin/bash

sudo docker build -t yelpanalysis .
sudo docker images
sudo docker run yelpanalysis

