#!/bin/sh
# Train OntoNotes first
python Train_STL_Sequence_Tagging.py -i input/OntoNotes_NW -p params/STL_Default_Param -ro results_emnlp -d STL_OntoNotes_NW 
wait 
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/ATIS/train.txt.ori -o data/ATIS_OntoNotes_Feature/train.txt.ori 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/ATIS/train.txt -o data/ATIS_OntoNotes_Feature/train.txt
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/ATIS/dev.txt -o data/ATIS_OntoNotes_Feature/dev.txt 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/ATIS/test.txt -o data/ATIS_OntoNotes_Feature/test.txt 
wait
# Train slot filling ATIS with NER as feature
python Train_STL_Sequence_Tagging.py -i input/ATIS_OntoNotes_Feature -ro results_emnlp -d STL_ATIS_OntoNotes_Feature -p params/STL_NER_OntoNotes_Feature_Param 
wait


python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/MIT_Restaurant/train.txt.ori -o data/MIT_Restaurant_OntoNotes_Feature/train.txt.ori 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/MIT_Restaurant/train.txt -o data/MIT_Restaurant_OntoNotes_Feature/train.txt
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/MIT_Restaurant/dev.txt -o data/MIT_Restaurant_OntoNotes_Feature/dev.txt 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5 -i data/MIT_Restaurant/test.txt -o data/MIT_Restaurant_OntoNotes_Feature/test.txt 
wait

python Train_STL_Sequence_Tagging.py -i input/MIT_Restaurant_OntoNotes_Feature -ro results_emnlp -d STL_MIT_Restaurant_OntoNotes_Feature -p params/STL_NER_OntoNotes_Feature_Param 
wait

python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5  -i data/MIT_Movie/train.txt.ori -o data/MIT_Movie_OntoNotes_Feature/train.txt.ori 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5  -i data/MIT_Movie/train.txt -o data/MIT_Movie_OntoNotes_Feature/train.txt
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5  -i data/MIT_Movie/dev.txt -o data/MIT_Movie_OntoNotes_Feature/dev.txt 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_OntoNotes_NW/models/OntoNotes_NW_model.h5  -i data/MIT_Movie/test.txt -o data/MIT_Movie_OntoNotes_Feature/test.txt 
wait
python Train_STL_Sequence_Tagging.py -i input/MIT_Movie_OntoNotes_Feature -ro results_emnlp -d STL_MIT_Movie_OntoNotes_Feature -p params/STL_NER_OntoNotes_Feature_Param