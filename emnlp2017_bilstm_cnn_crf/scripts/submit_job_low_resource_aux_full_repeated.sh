#!/bin/bash


:'
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 200 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_200_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 400 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_400_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 800 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_800_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
'
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 200 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Movie_NB_SENT_200_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 400 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Movie_NB_SENT_400_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 800 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_MIT_Movie_NB_SENT_800_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max


: '
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 10 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_ATIS_NB_SENT_10_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 20 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_ATIS_NB_SENT_20_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 40 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_ATIS_NB_SENT_40_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 80 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_ATIS_NB_SENT_80_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 100 -ner 1 -ner-name OntoNotes_NW -ro result_analyze -d MTL_ATIS_NB_SENT_100_OntoNotes_NW_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
'

: '
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_10_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_20_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_40_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_80_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_100_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_200_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_400_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_800_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait


python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_10_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_20_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_40_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_80_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_100_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_200_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_400_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_800_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait

python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_10_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 20 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_20_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 40 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_40_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 80 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_80_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 100 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_100_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_200_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_400_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_800_CONLL_NB_SENT_FULL_REPEATED -p params/MTL_Default_Param --batch-range max
wait
'

: '
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_NB_SENT_10_CONLL_NB_SENT_FULL -p params/MTL_Default_Param --batch-range max -e 2
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_NB_SENT_10_CONLL_NB_SENT_FULL -p params/MTL_Default_Param --batch-range max -e 2
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 10 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_NB_SENT_10_CONLL_NB_SENT_FULL -p params/MTL_Default_Param --batch-range max -e 2
wait
'
