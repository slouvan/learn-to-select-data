# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle, readCoNLL, remove_pkl_files,prepare_training_data
from util.CoNLL import dumpConll
from shutil import copyfile
import argparse

from util.preprocessing import perpareDataset, loadDatasetPickle, readCoNLL, remove_pkl_files,prepare_training_data

parser = argparse.ArgumentParser(description="Experiment Slot Filling")

parser.add_argument("-l", "--labeling-rate", dest="labeling_rate", help="Labeling Rate", metavar="N", type=float)

args = parser.parse_args()

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'CONLL_2003_NER':                            #Name of the dataset
        {'columns': {0:'tokens', 1:'CONLL_2003_BIO'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'CONLL_2003_BIO',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None,              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
         'proportion' : 0.1,
         'ori': True,
         'targetTask' : True}
}

labeling_rate = 0.0
if args.labeling_rate is not None :
    datasets['CONLL_2003_NER']['proportion'] = args.labeling_rate
else :
    datasets['CONLL_2003_NER']['proportion'] = 1

remove_pkl_files()
prepare_training_data(datasets)

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets, reducePretrainedEmbeddings=False)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/CONLL_NER_results.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
model.fit(epochs=25)



