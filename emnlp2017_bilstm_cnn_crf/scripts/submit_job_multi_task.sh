#!/bin/sh
python Train_Multitask.py -l 0.01
wait
python Train_Multitask.py -l 0.05
wait
#python Train_Multitask.py -l 0.1
#wait
#python Train_Multitask.py -l 0.2
#wait
#python Train_Multitask.py -l 0.4
#wait
#python Train_Multitask.py -l 0.6
#wait
#python Train_Multitask.py -l 0.8
#wait
#python Train_Multitask.py -l 1.0
#wait
