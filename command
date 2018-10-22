python bayes_opt_fbk.py --dynet-autobatch 1 -d small_data -m models/model -t ATIS --task slot_filling  -b random most-similar-examples -f similarity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 1


python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 50  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 50 --mtl-nb-sentence 400



# 5 Bayes iteration, 1 MTL epoch each
python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 1 --mtl-nb-sentence 200

python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 1 --mtl-nb-sentence 200


# EC2
python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 20  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 50 --mtl-nb-sentence 200


python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target ATIS --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_ATIS --mtl-epoch 1 --mtl-nb-sentence 200


# EC2
python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200


python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200

# Small CONLL 2003 small epoch BO small epoch MTL
python bayes_opt_fbk.py --dynet-autobatch 1 -d small_data -m models/model -t MIT_Restaurant --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200

# Full CONLL 2003 small epoch BO small epoch MTL
python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t MIT_Restaurant --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200

python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t MIT_Movie --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 20  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Movie --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Movie_200 --mtl-epoch 50 --mtl-nb-sentence 200

# Full CONLL 2003 BIG epoch BO BIG epoch MTL
python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t MIT_Restaurant --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 20  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 50 --mtl-nb-sentence 200

python bayes_opt_fbk.py --dynet-autobatch 1 -d small_data -m models/model -t MIT_Restaurant --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200



python bayes_opt_fbk.py --dynet-autobatch 1 -d data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Movie --mtl-aux CONLL_2003_NER --mtl-root-result /home/ubuntu/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Movie_200 --mtl-epoch 1 --mtl-nb-sentence 200


# SMALL RUN 
python bayes_opt_fbk.py --dynet-autobatch 1 -d small_data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Restaurant --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Restaurant_200 --mtl-epoch 1 --mtl-nb-sentence 200


python bayes_opt_fbk.py --dynet-autobatch 1 -d small_data -m models/model -t ATIS --task slot_filling -wv glove.6B.300d.txt -b random most-similar-examples -f similarity diversity --z-norm --num-iterations 5  --num-runs 1 --log-file logs/log -s CONLL_2003_NER --mtl-target MIT_Movie --mtl-aux CONLL_2003_NER --mtl-root-result /Users/slouvan/sandbox/learn-to-select-data/mtl_result --mtl-directory-name  MTL_MIT_Movie_200 --mtl-epoch 1 --mtl-nb-sentence 200
