#!/bin/sh
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro results_emnlp -diff-level 1 -d ATIS_ALL_NER_LOWER_LEVEL_LABEL_EMBEDDING -ner 1 -label-embedding label_embedding -e 25 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy most_similar -ro results_emnlp -diff-level 1 -d MIT_Restaurant_MOST_SIMILAR_NER_LOWER_LEVEL_LABEL_EMBEDDING -ner 1 -label-embedding label_embedding -e 25 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy all -ro results_emnlp  -d MIT_Movie_ALL_LABEL_EMBEDDING -label-embedding label_embedding -e 25 -p params/MTL_Default_Param