#!/bin/sh

# ATIS
python Train_STL_Sequence_Tagging.py -i input/ATIS -p params/STL_Default_Param -e 1  -d STL_TUNE_ATIS -t 1
wait

# MIT Restaurant
python Train_STL_Sequence_Tagging.py -i input/MIT_Restaurant -p params/STL_Default_Param -e 1  -d STL_TUNE_MIT_Restaurant -t 1
wait

# MIT Movie
python Train_STL_Sequence_Tagging.py -i input/MIT_Movie -p params/STL_Default_Param -e 1  -d STL_TUNE_MIT_Movie -t 1
wait

# MTL ATIS CoNLL
python Train_MTL_Sequence_Tagging.py -i input/ATIS_CoNLL  -p params/MTL_Default_Param -e 1  -d MTL_TUNE_ATIS_CONLL -t 1
wait

# MTL MIT Restaurant 
python Train_MTL_Sequence_Tagging.py -i input/MIT_Restaurant_CoNLL -p params/MTL_Default_Param -e 1  -d MTL_TUNE_MIT_Restaurant_CONLL -t 1
wait

# MTL MIT Movie CoNLL
python Train_MTL_Sequence_Tagging.py -i input/MIT_Movie_CoNLL  -p params/MTL_Default_Param -e 1  -d MTL_TUNE_MIT_Movie_CONLL -t 1
wait

# MTL ATIS OntoNotes_NW
python Train_MTL_Sequence_Tagging.py -i input/ATIS_OntoNotes_NW  -p params/MTL_Default_Param -e 1  -d MTL_TUNE_ATIS_OntoNotes_NW -t 1
wait

# MTL MIT Restaurant  OntoNotes NW
python Train_MTL_Sequence_Tagging.py -i input/MIT_Restaurant_OntoNotes_NW  -p params/MTL_Default_Param -e 1  -d MTL_TUNE_MIT_Restaurant_OntoNotes_NW -t 1
wait

# MTL MIT Movie OntoNotes NW
python Train_MTL_Sequence_Tagging.py -i input/MIT_Movie_OntoNotes_NW  -p params/MTL_Default_Param -e 1  -d MTL_TUNE_MIT_Movie_OntoNotes_NW -t 1
wait
