# This file contain an example how to perform multi-task learning on different levels.
# In the datasets variable, we specify two datasets: POS-tagging (unidep_pos) and conll2000_chunking.
# We pass a special parameter to the network (customClassifier), that allows that task are supervised at different levels.
# For the POS task, we use one shared LSTM layer followed by a softmax classifier. However, the chunking
# task uses the shared LSTM layer, then a task specific LSTM layer with 50 recurrent units, and then a CRF classifier.

from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle,prepare_training_data, read_dict, get_target_task, get_auxiliary_task
import argparse
from keras import backend as K

parser = argparse.ArgumentParser(description="Experiment Slot Filling")
parser.add_argument("-n", "--nb-sentence", dest="nb_sentence", help="Number of training sentence", type=int)
parser.add_argument("-ro", "--root-result", dest="root_dir_result", help="Root directory for results", default="results", type=str)
parser.add_argument("-d", "--directory-name", dest="directory_name", help="Directory Name", required = True, type=str)
parser.add_argument("-i", "--input", dest="input_dataset_conf", help="Input dataset configuration", required = True, type=str)
parser.add_argument("-p", "--param", dest="param_conf", help="Hyperparameters of the network", required=True, type=str)
parser.add_argument("-e", "--epoch", dest="nb_epoch", help="Number of epoch", default=50, type=int)
parser.add_argument("-t", "--tune", dest="tune", default=0, type=int)
parser.add_argument("-r", "--run", dest="nb_run", default =1, type = int)
args = parser.parse_args()

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
if os.path.exists("/".join([args.root_dir_result,args.directory_name])) :
    raise ValueError("The directory {} exists".format(args.directory_name))
else :
    print("The directory does not exist")

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
'''
datasets = {
    'unidep_pos':
        {'columns': {1:'tokens', 3:'POS'},
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None},
    'conll2000_chunking':
        {'columns': {0:'tokens', 2:'chunk_BIO'},
         'label': 'chunk_BIO',
         'evaluate': True,
         'commentSymbol': None},
}
'''

######################################################
#
# Data preprocessing
#
######################################################
datasets = read_dict(args.input_dataset_conf)
print("DATASET CONF {} {}".format(type(datasets), datasets))
target_task = get_target_task(datasets)
print("TARGET TASK {} {}".format(type(target_task), target_task))
aux_task = get_auxiliary_task(datasets)
print("AUX TASK {} {}".format(type(aux_task), aux_task))

prepare_training_data(datasets)


embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
#params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),'charEmbeddings': 'CNN',
#          'customClassifier': {'unidep_pos': ['Softmax'], 'conll2000_chunking': [('LSTM', 50), 'CRF']}}

# TODO Replace customClassifier dengan main task + auxiliary task
custom_classifier = {}
custom_classifier[target_task] = [('LSTM', 100), 'CRF']
for task in aux_task :
    custom_classifier[task] = ['CRF']

params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),'charEmbeddings': 'CNN',
          'customClassifier': custom_classifier}


model = BiLSTM(params)

model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults("/".join([args.root_dir_result, args.directory_name, "performance.out"]))  # Path to store performance scores for dev / test
model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name, "predictions", "[ModelName]_[Data].conll"])  # Path to store predictions
model.modelSavePath = "/".join([args.root_dir_result, args.directory_name, "models/[ModelName].h5"])  # Path to store models
model.fit(epochs=args.nb_epoch)



