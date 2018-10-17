#!/bin/sh
python Train_MultiTask_Different_Levels.py -d MIT_Movie_Different_Level_OntoNotes_NW -i input/MIT_Movie_OntoNotes_NW -p params/MTL_Default_Param -ro results_emnlp 
wait
python Train_MultiTask_Different_Levels.py -d MIT_Restaurant_Different_Level_OntoNotes_NW -i input/MIT_Restaurant_OntoNotes_NW -p params/MTL_Default_Param -ro results_emnlp 
wait
python Train_MultiTask_Different_Levels.py -d ATIS_Different_Level_OntoNotes_NW -i input/ATIS_OntoNotes_NW -p params/MTL_Default_Param -ro results_emnlp 
wait
python Train_MultiTask_Different_Levels.py -d MIT_Movie_Different_Level_CoNLL -i input/MIT_Movie_CoNLL -p params/MTL_Default_Param -ro results_emnlp 
wait
python Train_MultiTask_Different_Levels.py -d MIT_Restaurant_Different_Level_CoNLL -i input/MIT_Restaurant_CoNLL -p params/MTL_Default_Param -ro results_emnlp 
wait
python Train_MultiTask_Different_Levels.py -d ATIS_Different_Level_CoNLL -i input/ATIS_CoNLL -p params/MTL_Default_Param -ro results_emnlp 
wait