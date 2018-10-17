
from .constants import *
from .preprocessing import read_dict_data, set_target_task, perpareDataset, prepare_training_data, loadDatasetPickle
import sys

def construct_param(target_task, strategy= None, nb_sentence=None, ner=0, ner_name=None, diff_level = 0, nb_epoch=50, batch_range=None) :

    params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}
    aux_task = []
    if ner == 1:
        if ner_name is None:
            aux_task = aux_task + NERS
            if diff_level == 1:
                custom_classifier = {}  # Assuming NER always on the bottom
                custom_classifier[target_task[0]] = [('LSTM', 100), 'CRF']
                for task in aux_task:
                    if task in NERS:
                        custom_classifier[task] = ['CRF']
                    else:
                        custom_classifier[task] = [('LSTM', 100), 'CRF']

                params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN',
                          'customClassifier': custom_classifier}
            else:
                params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}
        else:
            for NER in NERS:
                if NER == ner_name:
                    print("{} is the NER aux task".format(NER))
                    aux_task.append(NER)
                    break
            if diff_level == 1:
                custom_classifier = {}  # Assuming NER always on the bottom
                custom_classifier[target_task[0]] = [('LSTM', 100), 'CRF']
                for task in aux_task:
                    if task in NERS:
                        custom_classifier[task] = ['CRF']
                    else:
                        custom_classifier[task] = [('LSTM', 100), 'CRF']

                params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN',
                          'customClassifier': custom_classifier}
            else:
                params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}

    return params

def construct_datasets(target_task, aux_task, nb_sentence = None) :

    datasets = read_dict_data([target_task] + aux_task)
    print(datasets)

    set_target_task(datasets, target_task)
    if nb_sentence is not None :
        datasets[target_task]['nb_sentence'] = nb_sentence

    embeddingsPath = 'komninos_english_embeddings.gz'  # Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    prepare_training_data(datasets)
    # :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
    pickleFile = perpareDataset(embeddingsPath, datasets, reducePretrainedEmbeddings=True)

    # Load the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    return embeddings, mappings, data, datasets


