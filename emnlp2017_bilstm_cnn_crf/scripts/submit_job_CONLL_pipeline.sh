#!/bin/sh
# Train CoNLL first
#python Train_STL_Sequence_Tagging.py -i input/CONLL_2003_NER -p params/STL_Default_Param -ro results_emnlp -d STL_CoNLL_2003
#wait 
#python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/ATIS/train.txt.ori -o data/ATIS_CoNLL_Feature/train.txt.ori 
#wait
#python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/ATIS/train.txt -o data/ATIS_CoNLL_Feature/train.txt
#wait
#python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/ATIS/dev.txt -o data/ATIS_CoNLL_Feature/dev.txt 
#wait
#python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/ATIS/test.txt -o data/ATIS_CoNLL_Feature/test.txt 
#wait
# Train slot filling ATIS with NER as feature
python Train_STL_Sequence_Tagging.py -i input/ATIS_CoNLL_Feature -ro results_emnlp -d STL_ATIS_CoNLL_Feature -p params/STL_NER_CoNLL_Feature_Param
wait


python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Restaurant/train.txt.ori -o data/MIT_Restaurant_CoNLL_Feature/train.txt.ori 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Restaurant/train.txt -o data/MIT_Restaurant_CoNLL_Feature/train.txt
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Restaurant/dev.txt -o data/MIT_Restaurant_CoNLL_Feature/dev.txt 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Restaurant/test.txt -o data/MIT_Restaurant_CoNLL_Feature/test.txt 
wait

python Train_STL_Sequence_Tagging.py -i input/MIT_Restaurant_CoNLL_Feature -ro results_emnlp -d STL_MIT_Restaurant_CoNLL_Feature -p params/STL_NER_CoNLL_Feature_Param
wait

python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Movie/train.txt.ori -o data/MIT_Movie_CoNLL_Feature/train.txt.ori 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Movie/train.txt -o data/MIT_Movie_CoNLL_Feature/train.txt
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Movie/dev.txt -o data/MIT_Movie_CoNLL_Feature/dev.txt 
wait
python RunModel_CoNLL_Format.py -m results_emnlp/STL_CoNLL_2003/models/CONLL_2003_NER_model.h5 -i data/MIT_Movie/test.txt -o data/MIT_Movie_CoNLL_Feature/test.txt 
wait
python Train_STL_Sequence_Tagging.py -i input/MIT_Movie_CoNLL_Feature -ro results_emnlp -d STL_MIT_Movie_CoNLL_Feature -p params/STL_NER_CoNLL_Feature_Param