#!/bin/sh

#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy most_similar -ro reproduce  -d ATIS_MOST_SIMILAR_NO_NER -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro reproduce  -d ATIS_ALL_NO_NER -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy most_similar -ro reproduce  -d ATIS_MOST_SIMILAR_NER_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro reproduce -d ATIS_ALL_NER_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait
#python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy none -ro reproduce  -d ATIS_NER_ONLY_SAME_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
#wait

python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy most_similar -diff-level 1 -ro reproduce  -diff-level 1 -d ATIS_MOST_SIMILAR_NER_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy all -ro reproduce -diff-level 1 -d ATIS_ALL_NER_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS  -strategy none -ro reproduce -diff-level 1 -d ATIS_NER_ONLY_DIFFERENT_LEVEL -ner 1  -e 25 -p params/MTL_Default_Param