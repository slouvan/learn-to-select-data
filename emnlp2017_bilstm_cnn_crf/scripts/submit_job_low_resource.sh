#!/bin/sh

:'
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_30 -i input/ATIS -p params/STL_Default_Param -n 30
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_40 -i input/ATIS -p params/STL_Default_Param -n 40
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_50 -i input/ATIS -p params/STL_Default_Param -n 50
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_60 -i input/ATIS -p params/STL_Default_Param -n 60
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_70 -i input/ATIS -p params/STL_Default_Param -n 70
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_80 -i input/ATIS -p params/STL_Default_Param -n 80
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_90 -i input/ATIS -p params/STL_Default_Param -n 90
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_ATIS_NB_SENT_100 -i input/ATIS -p params/STL_Default_Param -n 100
wait
'
:'
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_10 -i input/MIT_Restaurant -p params/STL_Default_Param -n 10
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_20 -i input/MIT_Restaurant -p params/STL_Default_Param -n 20
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_30 -i input/MIT_Restaurant -p params/STL_Default_Param -n 30
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_40 -i input/MIT_Restaurant -p params/STL_Default_Param -n 40
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_50 -i input/MIT_Restaurant -p params/STL_Default_Param -n 50
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_60 -i input/MIT_Restaurant -p params/STL_Default_Param -n 60
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_70 -i input/MIT_Restaurant -p params/STL_Default_Param -n 70
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_80 -i input/MIT_Restaurant -p params/STL_Default_Param -n 80
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_90 -i input/MIT_Restaurant -p params/STL_Default_Param -n 90
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Restaurant_NB_SENT_100 -i input/MIT_Restaurant -p params/STL_Default_Param -n 100
wait

python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_10 -i input/MIT_Movie -p params/STL_Default_Param -n 10
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_20 -i input/MIT_Movie -p params/STL_Default_Param -n 20
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_30 -i input/MIT_Movie -p params/STL_Default_Param -n 30
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_40 -i input/MIT_Movie -p params/STL_Default_Param -n 40
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_50 -i input/MIT_Movie -p params/STL_Default_Param -n 50
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_60 -i input/MIT_Movie -p params/STL_Default_Param -n 60
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_70 -i input/MIT_Movie -p params/STL_Default_Param -n 70
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_80 -i input/MIT_Movie -p params/STL_Default_Param -n 80
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_90 -i input/MIT_Movie -p params/STL_Default_Param -n 90
wait
python Train_STL_Sequence_Tagging.py -ro result_analyze -d STL_MIT_Movie_NB_SENT_100 -i input/MIT_Movie -p params/STL_Default_Param -n 100
'

python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_10 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_20 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 30 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_30 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_40 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 50 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_50 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 60 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_60 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 70 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_70 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_80 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 90 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_90 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_CONLL_NB_SENT_100 -p params/MTL_Default_Param
wait

python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_10 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_20 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 30 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_30 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_40 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 50 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_50 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 60 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_60 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 70 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_70 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_80 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 90 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_90 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_CONLL_NB_SENT_100 -p params/MTL_Default_Param
wait

python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_10 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_20 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 30 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_30 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_40 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 50 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_50 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 60 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_60 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 70 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_70 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_80 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 90 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_90 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_CONLL_NB_SENT_100 -p params/MTL_Default_Param
