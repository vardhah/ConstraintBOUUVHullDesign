#!/bin/bash


echo "Running freecad in Docker "

docker run  -it --rm --name tri -v ${PWD}/cad_sim:/home/ubuntu/butterfly vardhah/freecad:019 
echo "*** ALL DONE -- ALL DONE ***"
