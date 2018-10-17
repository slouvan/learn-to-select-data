#!/bin/sh

# ATIS NER ONLY
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy none -ro results_emnlp  -d ATIS_NER_ONLY_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy none -ro results_emnlp  -d MIT_Restaurant_NER_ONLY_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy none -ro results_emnlp  -d MIT_Movie_NER_ONLY_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy none -ro results_emnlp -diff-level 1 -d ATIS_NER_ONLY_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Restaurant_NER_ONLY_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Movie_NER_ONLY_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param


# ATIS NER ONLY DIFFERENT LEVEL

# MIT RESTAURANT NER ONLY DIFFERENT LEVEL

# MIT Movie NER ONLY DIFFERENT LEVEL


# LOW RESOURCE SCENARIOS

#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro results_emnlp -diff-level 1 -d ATIS_ALL_NER_LOWER_LEVEL_200 -ner 1  -e 25 -p params/MTL_Default_Param -n 200
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro results_emnlp -diff-level 1 -d ATIS_ALL_NER_LOWER_LEVEL_400 -ner 1  -e 25 -p params/MTL_Default_Param -n 400
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro results_emnlp -diff-level 1 -d ATIS_ALL_NER_LOWER_LEVEL_800 -ner 1  -e 25 -p params/MTL_Default_Param -n 800
#wait

#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy most_similar -ro results_emnlp -diff-level 1 -d MIT_Restaurant_MOST_SIMILAR_NER_LOWER_LEVEL_200 -ner 1  -e 25 -p params/MTL_Default_Param -n 200
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy most_similar -ro results_emnlp -diff-level 1 -d MIT_Restaurant_MOST_SIMILAR_NER_LOWER_LEVEL_400 -ner 1  -e 25 -p params/MTL_Default_Param -n 400
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy most_similar -ro results_emnlp -diff-level 1 -d MIT_Restaurant_MOST_SIMILAR_NER_LOWER_LEVEL_800 -ner 1  -e 25 -p params/MTL_Default_Param -n 800
#wait

#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy all -ro results_emnlp  -d MIT_Movie_ALL_200   -e 25 -p params/MTL_Default_Param -n 200
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy all -ro results_emnlp  -d MIT_Movie_ALL_400   -e 25 -p params/MTL_Default_Param -n 400
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy all -ro results_emnlp  -d MIT_Movie_ALL_800   -e 25 -p params/MTL_Default_Param -n 800


#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Restaurant_NER_ONLY_LOWER_LEVEL_200 -ner 1  -e 25 -p params/MTL_Default_Param -n 200
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Restaurant_NER_ONLY_LOWER_LEVEL_400 -ner 1  -e 25 -p params/MTL_Default_Param -n 400
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Restaurant_NER_ONLY_LOWER_LEVEL_800 -ner 1  -e 25 -p params/MTL_Default_Param -n 800
#wait

#python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Movie_NER_ONLY_LOWEL_LEVEL_200   -ner 1 -e 25 -p params/MTL_Default_Param -n 200
#wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy none -ro results_emnlp -diff-level 1 -d MIT_Movie_NER_ONLY_LOWER_LEVEL_400  -ner 1 -e 25 -p params/MTL_Default_Param -n 400
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie  -strategy none -ro results_emnlp  -diff-level 1 -d MIT_Movie_NER_ONLY_LOWER_LEVEL_800  -ner 1 -e 25 -p params/MTL_Default_Param -n 800