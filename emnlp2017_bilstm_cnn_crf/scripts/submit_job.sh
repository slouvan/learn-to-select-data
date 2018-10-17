#!/bin/sh
python Train_MIT_Movie.py -n 200 -d SingleTask_MIT_Movie_200
wait
python Train_MIT_Movie.py -n 400 -d SingleTask_MIT_Movie_400
wait
python Train_MIT_Movie.py -n 800 -d SingleTask_MIT_Movie_800
wait
python Train_MultiTask_MIT_Movie.py -n 200 -d MultiTask_MIT_Movie_200
wait
python Train_MultiTask_MIT_Movie.py -n 400 -d MultiTask_MIT_Movie_400
wait
python Train_MultiTask_MIT_Movie.py -n 800 -d MultiTask_MIT_Movie_800
wait