# This file contain an example how to perform multi-task learning using the
# BiLSTM-CNN-CRF implementation.
# In the datasets variable, we specify two datasets: POS-tagging (unidep_pos) and conll2000_chunking.
# The network will then train jointly on both datasets.
# The network can on more datasets by adding more entries to the datasets dictionary.

from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
import argparse
from util.preprocessing import perpareDataset, loadDatasetPickle, readCoNLL, remove_pkl_files,prepare_training_data, read_dict, get_target_task
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
#datasets = {
#    'unidep_pos':
#        {'columns': {1:'tokens', 3:'POS'},
#         'label': 'POS',
#         'evaluate': True,
#         'commentSymbol': None},
#    'conll2000_chunking':
#        {'columns': {0:'tokens', 2:'chunk_BIO'},
#         'label': 'chunk_BIO',
#         'evaluate': True,
#         'commentSymbol': None},
#}

######################################################
#
# Data preprocessing
#
######################################################
datasets = read_dict(args.input_dataset_conf)
print("DATASET CONF {} {}".format(type(datasets), datasets))
target_task = get_target_task(datasets)
print("TARGET TASK {} {}".format(type(target_task), target_task))

# get the key where the dataset is the target task
if args.nb_sentence is not None :
    datasets[target_task]['nb_sentence'] = args.nb_sentence


#remove_pkl_files()
prepare_training_data(datasets)

embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets, reducePretrainedEmbeddings=True)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = read_dict(args.param_conf)
print("{} {}".format(type(params), params))

if args.tune == 0:
    if args.nb_run == 1:
        model = BiLSTM(params)
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data, mainModelName=target_task)  # KHUSUS MULTITSAK

        model.storeResults("/".join([args.root_dir_result,args.directory_name,"performance.out"])) #Path to store performance scores for dev / test
        model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name,"predictions","[ModelName]_[Data].conll"]) #Path to store predictions
        model.modelSavePath = "/".join([args.root_dir_result,args.directory_name,"models/[ModelName].h5"]) #Path to store models

        model.fit(epochs=args.nb_epoch)
        model.saveParams("/".join([args.root_dir_result,args.directory_name,"param"]))
    else :

        for current_run in range(1, args.nb_run + 1):
            model = BiLSTM(params)
            model.setMappings(mappings, embeddings)
            model.setDataset(datasets, data, mainModelName=target_task)  # KHUSUS MULTITSAK

            model.storeResults("/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "performance.out"]))  # Path to store performance scores for dev / test
            model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "predictions", "[ModelName]_[Data].conll"])  # Path to store predictions
            model.modelSavePath = "/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "models/[ModelName].h5"])  # Path to store models

            model.fit(epochs=args.nb_epoch)
            model.saveParams("/".join([args.root_dir_result, args.directory_name+ "_"+str(current_run), "param"]))

else :
    print("Tuning")
    drop_out_tuning = [0.25, 0.35, 0.45, 0.5]
    for current_drop_out in drop_out_tuning :
        params['dropout'] = (current_drop_out, current_drop_out)
        model = BiLSTM(params)
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data, mainModelName=target_task)  # KHUSUS MULTITSAK

        model.storeResults("/".join([args.root_dir_result, args.directory_name, "performance.out"]))  # Path to store performance scores for dev / test
        model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name, "predictions", "[ModelName]_[Data].conll"])  # Path to store predictions
        model.modelSavePath = "/".join([args.root_dir_result, args.directory_name, "models/[ModelName].h5"])  # Path to store models
        model.fit(epochs=args.nb_epoch)
        model.saveParams("/".join([args.root_dir_result, args.directory_name, "param"]))
        model.saveParamTuningResults("/".join([args.root_dir_result, args.directory_name, "tuning_results"]))


