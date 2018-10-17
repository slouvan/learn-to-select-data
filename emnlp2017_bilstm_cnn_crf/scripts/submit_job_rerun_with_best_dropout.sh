#!/bin/sh
python Train_STL_Sequence_Tagging.py -i input/MIT_Movie -d STL_MIT_Movie_BestDropout_12_July -p params/STL_MIT_Movie_BestParam
wait
python Train_MTL_Sequence_Tagging.py -i input/ATIS_CoNLL -d MTL_ATIS_CoNLL_BestDropout_12_July -p params/MTL_ATIS_CoNLL_BestParam
wait
python Train_MTL_Sequence_Tagging.py -i input/MIT_Movie_CoNLL -d MTL_MIT_Movie__CoNLL_BestDropout_12_July -p params/MTL_MIT_Movie_CoNLL_BestParam
wait
python Train_MTL_Sequence_Tagging.py -i input/MIT_Restaurant_CoNLL -d MTL_MIT_Restaurant_CoNLL_BestDropout_12_July -p params/MTL_MIT_Restaurant_CoNLL_BestParam
wait
python Train_MTL_Sequence_Tagging.py -i input/ATIS_OntoNotes_NW -d MTL_ATIS_OntoNotes_NW_BestDropout_12_July -p params/MTL_MIT_Movie_OntoNotes_NW_BestParam
wait
python Train_MTL_Sequence_Tagging.py -i input/MIT_Restaurant_OntoNotes_NW -d MTL_MIT_Restaurant_OntoNotes_NW_BestDropout_12_July -p params/MTL_MIT_Restaurant_OntoNotes_NW_BestParam
