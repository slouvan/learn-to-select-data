from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import gzip
import os.path
import nltk
import logging
from nltk import FreqDist
from .CoNLL import dumpConll
from .WordEmbeddings import wordNormalize
from .CoNLL import readCoNLL
from shutil import copyfile
from collections import defaultdict
import scipy
import sys
from .constants import DATA_DIR, INPUT_DIR, TASKS
import math
from scipy.linalg import svd
from constants import MTL_DATA, MTL_PKL
from collections import defaultdict

if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
    from io import open


def remove_pkl_files():
    import os
    import glob

    files = glob.glob('pkl/*')
    for f in files:
        print("REMOVING pk file")
        os.remove(f)

def retag_sentences(sentences, relevant_tags, column, task):
    counter = 0
    filtered_sentence = []
    assert("BIO" in column)
    for idx, sentence in enumerate(sentences):
        original_labels = sentence[column]
        retag_labels = []
        retag_happened = False
        for label in original_labels:
            if get_label_name(label) != "O" and get_label_name(label) not in relevant_tags[task]:
                #print("Retag happening from {} to {} in task {}, relevant tags are : {}".format(label, "O", task, list(relevant_tags[task])))
                retag_labels.append("O")
                retag_happened = True
            else :
                retag_labels.append(label)
        sentences[idx][column] = retag_labels
        assert (len(original_labels) == len(retag_labels))
        filtered_sentence.append(sentence)

    assert(len(sentences) == len(filtered_sentence))
    return filtered_sentence


def prepare_training_data(datasets, filter_tags=None) :
    data_folder = MTL_DATA
    for dataset_name, props in datasets.items():
            sentences = None
            from numpy.random import shuffle
            if props['ori'] and os.path.isfile(os.path.join(data_folder,dataset_name,'train.txt.ori')) and props['nb_sentence'] is None:
                sentences = readCoNLL(os.path.join(data_folder, dataset_name, 'train.txt.ori'), props['columns'])
                if filter_tags is not None and dataset_name != get_target_task(datasets):
                    print("Retagging {}".format(dataset_name))
                    sentences = retag_sentences(sentences, filter_tags, props['columns'][1], dataset_name)
                    dumpConll(os.path.join(data_folder, dataset_name, 'train.txt'), sentences, props['columns'])
                else :
                    copyfile(os.path.join(data_folder, dataset_name, 'train.txt.ori'), os.path.join(data_folder, dataset_name, 'train.txt'))
            else:
                sentences = readCoNLL(os.path.join(data_folder, dataset_name, 'train.txt.ori'), props['columns'])
                if filter_tags is not None and dataset_name != get_target_task(datasets):
                    print("Retagging")
                    sentences = retag_sentences(sentences, filter_tags, props['columns'][1], dataset_name)
                np.random.seed(13)
                shuffled_indices = np.random.choice(len(sentences), props['nb_sentence'])
                sentences = np.asarray(sentences)[shuffled_indices].tolist()
                #print(shuffled_indices)
                dumpConll(os.path.join(data_folder, dataset_name, 'train.txt'), sentences, props['columns'])

            print("Total number of sentence in  {}  is {}".format(dataset_name, len(sentences)))


def perpareDataset(embeddingsPath, datasets, frequencyThresholdUnknownTokens=50, labeling_rate = 1.0,reducePretrainedEmbeddings=False, valTransformations=None, padOneTokenSentence=True):
    """
    Reads in the pre-trained embeddings (in text format) from embeddingsPath and prepares those to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least frequencyThresholdUnknownTokens times
    
    # Arguments:
        embeddingsPath: Full path to the pre-trained embeddings file. File must be in text format.
        datasetFiles: Full path to the [train,dev,test]-file
        frequencyThresholdUnknownTokens: Unknown words are added, if they occure more than frequencyThresholdUnknownTokens times in the train set
        reducePretrainedEmbeddings: Set to true, then only the embeddings needed for training will be loaded
        valTransformations: Column specific value transformations
        padOneTokenSentence: True to pad one sentence tokens (needed for CRF classifier)
    """
    embeddingsName = os.path.splitext(embeddingsPath)[0]
    pklName = "_".join(sorted(datasets.keys()) + [embeddingsName])
    #outputPath = 'pkl/' + pklName + '_'+str(labeling_rate)+'.pkl'
    outputPath = os.path.join(MTL_PKL, pklName + '_' + str(labeling_rate) + '.pkl')
    #if os.path.isfile(outputPath):
    #    logging.info("Using existent pickle file: %s" % outputPath)
    #    return outputPath

    casing2Idx = getCasingVocab()
    embeddings, word2Idx = readEmbeddings(embeddingsPath, datasets, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings)
    
    mappings = {'tokens': word2Idx, 'casing': casing2Idx}
    pklObjects = {'embeddings': embeddings, 'mappings': mappings, 'datasets': datasets, 'data': {}}

    for datasetName, dataset in datasets.items():
        datasetColumns = dataset['columns']
        commentSymbol = dataset['commentSymbol']

        trainData = os.path.join(MTL_DATA, '%s/train.txt' % datasetName)
        devData = os.path.join(MTL_DATA, '%s/dev.txt' % datasetName)
        testData = os.path.join(MTL_DATA, '%s/test.txt' % datasetName)
        paths = [trainData, devData, testData]

        logging.info(":: Transform "+datasetName+" dataset ::")
        pklObjects['data'][datasetName] = createPklFiles(paths, mappings, datasetColumns, commentSymbol, valTransformations, padOneTokenSentence)

    
    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()

    return pklObjects['embeddings'], pklObjects['mappings'], pklObjects['data']



def readEmbeddings(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings):
    """
    Reads the embeddingsPath.
    :param embeddingsPath: File path to pretrained embeddings
    :param datasetName:
    :param datasetFiles:
    :param frequencyThresholdUnknownTokens:
    :param reducePretrainedEmbeddings:
    :return:
    """
    # Check that the embeddings file exists
    if not os.path.isfile(embeddingsPath):
        if embeddingsPath in ['komninos_english_embeddings.gz', 'levy_english_dependency_embeddings.gz', 'reimers_german_embeddings.gz']:
            getEmbeddings(embeddingsPath)
        else:
            print("The embeddings file %s was not found" % embeddingsPath)
            exit()

    logging.info("Generate new embeddings files for a dataset")

    neededVocab = {}
    if reducePretrainedEmbeddings:
        logging.info("Compute which tokens are required for the experiment")

        def createDict(filename, tokenPos, vocab):
            for line in open(filename):
                if line.startswith('#'):
                    continue
                splits = line.strip().split()
                if len(splits) > 1:
                    word = splits[tokenPos]
                    wordLower = word.lower()
                    wordNormalized = wordNormalize(wordLower)

                    vocab[word] = True
                    vocab[wordLower] = True
                    vocab[wordNormalized] = True

        for key, dataset in datasetFiles.items():
            print(dataset)

            dataColumnsIdx = {y: x for x, y in dataset['columns'].items()}

            tokenIdx = dataColumnsIdx['tokens']
            #datasetPath = 'data/%s/' % dataset['name']
            datasetPath = '%s/' % key

            for dataset in ['train.txt', 'dev.txt', 'test.txt']:
                createDict(os.path.join(MTL_DATA, datasetPath, dataset), tokenIdx, neededVocab)

    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")

    embeddingsDimension = None

    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            np.random.seed(13)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    # Extend embeddings file with new tokens
    def createFD(filename, tokenIndex, fd, word2Idx):
        for line in open(filename):
            if line.startswith('#'):
                continue

            splits = line.strip().split()

            if len(splits) > 1:
                word = splits[tokenIndex]
                wordLower = word.lower()
                wordNormalized = wordNormalize(wordLower)

                if word not in word2Idx and wordLower not in word2Idx and wordNormalized not in word2Idx:
                    fd[wordNormalized] += 1

    if frequencyThresholdUnknownTokens != None and frequencyThresholdUnknownTokens >= 0:
        fd = nltk.FreqDist()
        for datasetName, datasetFile in datasetFiles.items():
            dataColumnsIdx = {y: x for x, y in datasetFile['columns'].items()}
            tokenIdx = dataColumnsIdx['tokens']
            #datasetPath = 'data/%s/' % datasetName
            datasetPath = os.path.join(MTL_DATA, '%s'% datasetName, 'train.txt')
            createFD(datasetPath, tokenIdx, fd, word2Idx)

        addedWords = 0
        for word, freq in fd.most_common(10000):
            if freq < frequencyThresholdUnknownTokens:
                break

            addedWords += 1
            word2Idx[word] = len(word2Idx)
            np.random.seed(13)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

            assert (len(word2Idx) == len(embeddings))

        logging.info("Added words: %d" % addedWords)
    embeddings = np.array(embeddings)

    return embeddings, word2Idx


def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)

def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def createMatrices(sentences, mappings, padOneTokenSentence, padMaxSentlength = False):
    data = []
    numTokens = 0
    numUnknownTokens = 0    
    missingTokens = FreqDist()
    paddedSentences = 0

    maxSentLength = max([len(sentence['tokens']) for sentence in sentences])
    for sentence in sentences:
        for entry in sentence['tokens'] :
            pass
            #print("ENTRY : {}".format(entry))
    print("Max sent length is : {}".format(maxSentLength))

    #print("Sentence max length is : {}".format(maxSentLength))
    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:
                #print(" Entry : {}".format(entry))
                if mapping.lower() == 'tokens':
                    numTokens += 1
                    idx = str2Idx['UNKNOWN_TOKEN']
                    
                    if entry in str2Idx:
                        idx = str2Idx[entry]
                    elif entry.lower() in str2Idx:
                        idx = str2Idx[entry.lower()]
                    elif wordNormalize(entry) in str2Idx:
                        idx = str2Idx[wordNormalize(entry)]
                    else:
                        numUnknownTokens += 1    
                        missingTokens[wordNormalize(entry)] += 1
                        
                    row['raw_tokens'].append(entry)
                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)
                #if mapping.lower() == 'characters' :
                #    print("Mapping {} ".format(row['characters']))

        if len(row['tokens']) == 1 and padOneTokenSentence:
            paddedSentences += 1
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    #row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    #print("Masuk sini")
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)


        if padMaxSentlength and len(row['tokens']) < maxSentLength :
            nb_iteration_for_padding = maxSentLength - len(row['tokens'])
            print("Padding stuff")
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    for i in range(0, nb_iteration_for_padding) :
                        row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                        #row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    for i in range(0, nb_iteration_for_padding):
                        row['characters'].append([0])
                        #print("Masuk sini character")
                else:
                    #print("Masuk sini yang lain")
                    for i in range(0, nb_iteration_for_padding):
                        row[mapping].append(0)

        data.append(row)

    if numTokens > 0:
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens/float(numTokens)*100))
    #print("DATA : {}".format(data))
    #os._exit(2)
    return data
    
  
  
def createPklFiles(datasetFiles, mappings, cols, commentSymbol, valTransformation, padOneTokenSentence):
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol, valTransformation)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol, valTransformation)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol, valTransformation)    
   
    extendMappings(mappings, trainSentences+devSentences+testSentences)

    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset

    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    
    addCharInformation(testSentences)   
    addCasingInformation(testSentences)

    logging.info(":: Create Train Matrix ::")
    trainMatrix = createMatrices(trainSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Dev Matrix ::")

    devMatrix = createMatrices(devSentences, mappings, padOneTokenSentence)
    logging.info(":: Create Test Matrix ::")
    testMatrix = createMatrices(testSentences, mappings, padOneTokenSentence)

    
    data = {
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def extendMappings(mappings, sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens') #No need to map tokens

    for sentence in sentences:
        for name in sentenceKeys:
            if name not in mappings:
                mappings[name] = {'O':0} #'O' is also used for padding

            for item in sentence[name]:              
                if item not in mappings[name]:
                    mappings[name][item] = len(mappings[name])



    

def getEmbeddings(name):
    if not os.path.isfile(name):
        download("https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/"+name)



def getLevyDependencyEmbeddings():
    """
    Downloads from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    the dependency based word embeddings and unzips them    
    """ 
    if not os.path.isfile("levy_deps.words.bz2"):
        print("Start downloading word embeddings from Levy et al. ...")
        os.system("wget -O levy_deps.words.bz2 http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2")
    
    print("Start unzip word embeddings ...")
    os.system("bzip2 -d levy_deps.words.bz2")

def getReimersEmbeddings():
    """
    Downloads from https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/
    embeddings for German
    """
    if not os.path.isfile("2014_tudarmstadt_german_50mincount.vocab.gz"):
        print("Start downloading word embeddings from Reimers et al. ...")
        os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2014_german_embeddings/2014_tudarmstadt_german_50mincount.vocab.gz")
    
   

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
    from urllib.request import urlretrieve
else:
    import urllib2
    import urlparse
    from urllib import urlretrieve


def download(url, destination=os.curdir, silent=False):
    filename = os.path.basename(urlparse.urlparse(url).path) or 'downloaded.file'

    def get_size():
        meta = urllib2.urlopen(url).info()
        meta_func = meta.getheaders if hasattr(
            meta, 'getheaders') else meta.get_all
        meta_length = meta_func('Content-Length')
        try:
            return int(meta_length[0])
        except:
            return 0

    def kb_to_mb(kb):
        return kb / 1024.0 / 1024.0

    def callback(blocks, block_size, total_size):
        current = blocks * block_size
        percent = 100.0 * current / total_size
        line = '[{0}{1}]'.format(
            '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
        sys.stdout.write(
            status.format(
                percent, line, kb_to_mb(current), kb_to_mb(total_size)))

    path = os.path.join(destination, filename)

    logging.info(
        'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
    try:
        (path, headers) = urlretrieve(url, path, None if silent else callback)
    except:
        os.remove(path)
        raise Exception("Can't download {0}".format(path))
    else:
        print()
        logging.info('Downloaded to: {0}'.format(path))

    return path

'''
    The input for input data conf is following the existing format in dictionary format.
    This function load the configuration to a dictionary
    Resource : https://stackoverflow.com/questions/9314824/python-create-dictionary-from-text-file-thats-in-dictionary-format
'''
def read_dict(filename) :
    dict_from_file = None
    with open(filename, "r") as f:
        dict_from_file = eval(f.read())

    return dict_from_file

'''
    Returns the dataset name of the target task
'''
def get_target_task(datasets) :
    names = []
    for key, item in datasets.items():
        if datasets[key]['evaluate'] :
            names.append(key)
    if len(names) > 1:
        raise("Error in the dataset configuration. There are more than 1 target task")

    return names[0]

def get_auxiliary_task(datasets):
    names = []
    for key, item in datasets.items():
        if not datasets[key]['evaluate']:
            names.append(key)


    return names

def build_vocab_from_domains(domains):
    word2idx = {}
    print(len(word2idx))
    for domain in domains :
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain,"train.txt.ori"), {0:'tokens'})
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            for token in tokens :
                if token.lower() not in word2idx.keys() :
                    word2idx[token.lower()] = len(word2idx)

    return word2idx

def build_all_domain_term_dist(domains, word2idx) :
    domain_to_term_dist = {}
    for domain in domains :
        domain_to_term_dist[domain] = create_term_dist(domain, word2idx)

    return domain_to_term_dist

def create_term_dist(domain, word2idx) :
    term_dist = np.zeros(len(word2idx))
    sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain,"train.txt.ori"), {0: 'tokens'})
    for sentence_idx in range(len(sentences_in_domain)) :
        tokens = sentences_in_domain[sentence_idx]['tokens']
        for token in tokens :
            term_dist[word2idx[token.lower()]] += 1

    term_dist = term_dist / np.sum(term_dist)

    return term_dist

def get_most_similar_domain(target_domain, domains, domain_to_term_dist):

    best_similarity_score , most_similar_domain = 0, None
    for source_domain in domains:
        if source_domain != target_domain:
            similarity_score = get_similarity_score(domain_to_term_dist[source_domain], domain_to_term_dist[target_domain])
            if similarity_score > best_similarity_score :
                best_similarity_score = similarity_score
                most_similar_domain = source_domain

    return most_similar_domain, best_similarity_score


# From Sebastian Ruder
# https://github.com/sebastianruder/learn-to-select-data/blob/5a5fe2428a465d6d0a7a6ffbd47c89baffc09531/similarity.py

def get_similarity_score(repr1, repr2) :
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim

def read_dict_data(domain_names) :
    dict_from_file = None
    all_data = {}
    for domain in domain_names:
        print("read_dict_data from {} ".format(domain))
        with open(os.path.join(INPUT_DIR, domain), "r") as f:
            dict_from_file = eval(f.read())
            all_data[domain] = dict_from_file[domain]
    return all_data


def set_target_task(datasets, target_task) :
    keys = list(datasets.keys())
    for key in keys:
        if key == target_task:
            datasets[key]['evaluate'] = True
        else :
            datasets[key]['evaluate'] = False

def get_label_name(token_label) :
    if token_label == "O" :
        return "O"
    if token_label.startswith("B-") and token_label.startswith("I-"):
        raise ValueError("Wrong annotation format")

    if token_label.startswith("B-"):
        fields = token_label.split("B-")
        return fields[1]
    elif token_label.startswith("I-"):
        fields = token_label.split("I-")
        return fields[1]


def build_indexes_from_domains(domains):
    word2idx = {}
    label2idx = {}

    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        print(" Number of sentences in {} : {}".format(domain, len(sentences_in_domain)))
        for sentence_idx in range(len(sentences_in_domain)):
            #if sentence_idx % 1000 == 0:
            #    print("{}".format(sentence_idx))
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                if tokens[token_idx].lower() not in word2idx.keys():
                    word2idx[tokens[token_idx].lower()] = len(word2idx)
                label = get_label_name(labels[token_idx])
                if label not in label2idx.keys():
                    label2idx[label] = len(label2idx)

    idx2label = { v : k for k, v in label2idx.items()}
    idx2word  = { v : k for k, v in word2idx.items()}

    idx = {'word2idx' : word2idx, 'idx2word': idx2word, 'label2idx' : label2idx, 'idx2label' : idx2label}
    print("{} {}".format(len(word2idx), len(label2idx)))
    return idx

def build_matrix_from_domains(domains, idx) :

    label_count = defaultdict(int)
    word_count = defaultdict(int)
    label_word_count = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                label = get_label_name(labels[token_idx])
                label_count[label] += 1
                word_count[token] += 1
                label_word_count[idx['word2idx'][token], idx['label2idx'][label]] += 1

    original_matrix = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))
    for i in range(label_word_count.shape[0]):
        for j in range(label_word_count.shape[1]):
            original_matrix[i, j] = label_word_count[i,j] / math.sqrt(label_count[idx['idx2label'][i]] * word_count[idx['idx2word'][j]])

    return original_matrix

def get_svd(matrix) :
    return svd(matrix)

def build_vocab(domains):
    word_count = defaultdict(int)
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        print(" Number of sentences in {} : {}".format(domain, len(sentences_in_domain)))
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                word_count[token] += 1
    return word_count


def build_indexes_from_domains(domains, word_count, threshold=5):
    word2idx = {}
    label2idx = {}
    domain2label = {}
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        print(" Number of sentences in {} : {}".format(domain, len(sentences_in_domain)))
        labels_in_domain = set()
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']

            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                label = get_label_name(labels[token_idx])
                if label not in label2idx.keys() and label != "O":
                    label2idx[label] = len(label2idx)
                    labels_in_domain.add(label)
                if token not in word2idx.keys() and word_count[token] >= threshold:
                    word2idx[token] = len(word2idx)
        domain2label[domain] = list(labels_in_domain)

    idx2label = {v: k for k, v in label2idx.items()}
    idx2word  = {v: k for k, v in word2idx.items()}

    idx = {'word2idx': word2idx, 'idx2word': idx2word, 'label2idx': label2idx, 'idx2label': idx2label, 'domain2label': domain2label}
    print("Word : {} Label :{}".format(len(word2idx), len(label2idx)))
    return idx


def build_matrix_from_domains(domains, idx, word_count):
    label_count = defaultdict(int)
    label_word_count = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                label = get_label_name(labels[token_idx])
                if token in idx['word2idx'].keys() and label != "O" and label != "LAW":
                    label_count[label] += 1
                    label_word_count[idx['label2idx'][label], idx['word2idx'][token]] += 1

    original_matrix = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))

    print(label_count)
    for i in range(label_word_count.shape[0]):
        for j in range(label_word_count.shape[1]):
            if math.sqrt(label_count[idx['idx2label'][i]] * word_count[idx['idx2word'][j]]) != 0:
                original_matrix[i, j] = label_word_count[i, j] / math.sqrt(
                    label_count[idx['idx2label'][i]] * word_count[idx['idx2word'][j]])

    return original_matrix


def build_vocab(domains):
    word_count = defaultdict(int)
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        print(" Number of sentences in {} : {}".format(domain, len(sentences_in_domain)))
        for sentence_idx in range(len(sentences_in_domain)):
            # if sentence_idx % 1000 == 0:
            #    print("{}".format(sentence_idx))
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                word_count[token] += 1
    return word_count


def build_indexes_from_domains(domains, word_count, threshold=5):
    word2idx = {}
    label2idx = {}
    domain2label = {}
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        print(" Number of sentences in {} : {}".format(domain, len(sentences_in_domain)))
        labels_in_domain = set()
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                label = get_label_name(labels[token_idx])
                if label not in label2idx.keys() and label != "O":
                    label2idx[label] = len(label2idx)
                    labels_in_domain.add(label)
                if token not in word2idx.keys() and word_count[token] >= threshold:
                    word2idx[token] = len(word2idx)
        domain2label[domain] = list(labels_in_domain)

    idx2label = {v: k for k, v in label2idx.items()}
    idx2word = {v: k for k, v in word2idx.items()}

    idx = {'word2idx': word2idx, 'idx2word': idx2word, 'label2idx': label2idx, 'idx2label': idx2label,
           'domain2label': domain2label}
    print("Word : {} Label :{}".format(len(word2idx), len(label2idx)))
    return idx


from collections import defaultdict
import numpy as np
import math


def build_matrix_from_domains(domains, idx, word_count, k=50):
    label_count = defaultdict(int)
    label_word_count = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))
    for domain in domains:
        sentences_in_domain = readCoNLL(os.path.join(DATA_DIR, domain, "train.txt.ori"), {0: 'tokens', 1: 'labels'})
        for sentence_idx in range(len(sentences_in_domain)):
            tokens = sentences_in_domain[sentence_idx]['tokens']
            labels = sentences_in_domain[sentence_idx]['labels']
            for token_idx in range(len(tokens)):
                token = tokens[token_idx].lower()
                label = get_label_name(labels[token_idx])
                if token in idx['word2idx'].keys() and label != "O" and label != "LAW":
                    label_count[label] += 1
                    label_word_count[idx['label2idx'][label], idx['word2idx'][token]] += 1

    original_matrix = np.zeros((len(idx['label2idx']), len(idx['word2idx'])))

    for i in range(label_word_count.shape[0]):
        for j in range(label_word_count.shape[1]):
            if math.sqrt(label_count[idx['idx2label'][i]] * word_count[idx['idx2word'][j]]) != 0:
                original_matrix[i, j] = label_word_count[i, j] / math.sqrt(
                    label_count[idx['idx2label'][i]] * word_count[idx['idx2word'][j]])

    from scipy.linalg import svd
    M1, M2, M3 = svd(original_matrix)
    row_sums = M1.sum(axis=1)
    normalized_matrix = M1 / row_sums[:, np.newaxis]

    ranked_k = normalized_matrix[:, :k]
    return ranked_k


def get_label_mapping(domain1, domain2, matrix, idxs):
    for label1 in idxs['domain2label'][domain1]:
        highest_sim_score = -1000000000000
        nearest_neighbor = None
        for label2 in idxs['domain2label'][domain2]:
            score = get_similarity(ranked_k[idxs['label2idx'][label1]], ranked_k[idxs['label2idx'][label2]])
            if score > highest_sim_score:
                highest_sim_score = score
                nearest_neighbor = label2
        print("The nearest neighbor for {} is {} with the score of {}".format(label1, nearest_neighbor,
                                                                              highest_sim_score))


from scipy.spatial.distance import cosine, euclidean


def get_similarity(repr1, repr2):
    return 1 - cosine(repr1, repr2)


def get_distance(repr1, repr2):
    return euclidean(repr1, repr2)


def get_nearest_labels(target_task, aux_tasks, matrix, idxs, sim_threshold=0.1):
    nearest_labels = {}

    for aux_task in aux_tasks:
        unique_labels = set()
        for label1 in idxs['domain2label'][target_task]:
            highest_sim_score = -1000000000000
            nearest_neighbor = None
            for label2 in idxs['domain2label'][aux_task]:
                if not np.any(matrix[idxs['label2idx'][label1]]) or not np.any(matrix[idxs['label2idx'][label2]]):
                    continue
                score = get_similarity(matrix[idxs['label2idx'][label1]], matrix[idxs['label2idx'][label2]])
                if score > highest_sim_score:
                    highest_sim_score = score
                    nearest_neighbor = label2
            # print("The nearest neighbor for {} is {} with the score of {}".format(label1, nearest_neighbor, highest_sim_score))
            if highest_sim_score >= sim_threshold:
                unique_labels.add(nearest_neighbor)
        nearest_labels[aux_task] = unique_labels
        #print("Nearest labels from {}  is {}".format(aux_task, str(unique_labels)))

    return nearest_labels

def compute_label_embeddings (target_task, aux_tasks):
    word_count = build_vocab(target_task + aux_tasks)
    idxs = build_indexes_from_domains(target_task + aux_tasks, word_count)
    matrix = build_matrix_from_domains(target_task + aux_tasks, idxs, word_count)
    return get_nearest_labels(target_task[0], aux_tasks, matrix,idxs, sim_threshold=0.1)