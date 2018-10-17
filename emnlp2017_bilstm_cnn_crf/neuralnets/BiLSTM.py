"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging used for multi-task learning.

Author: Nils Reimers
License: Apache-2.0
"""

from __future__ import print_function

from util import BIOF1Validation
import numpy as np
np.random.seed(1337)
import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging
from collections import defaultdict
from .keraslayers.ChainCRF import ChainCRF
import datetime
import json
os.environ['PYTHONHASHSEED'] = '0'

class BiLSTM:
    def __init__(self, params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None
        self.predictionSavePath = None
        self.earlyStoppingTargetTask = None
        self.customizedAlternate = None
        self.current_dev_prediction = None
        self.current_test_prediction = None
        # Hyperparameters for the network
        defaultParams = {'dropout': (0.5,0.5), 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam',
                         'charEmbeddings': None, 'charEmbeddingsSize': 30, 'charFilterSize': 30, 'charFilterLength': 3, 'charLSTMSize': 25, 'maxCharLength': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 10, 'miniBatchSize': 32,
                         'featureNames': ['tokens', 'casing'], 'addFeatureDimensions': 10}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

        self.final_max_dev_score = 0
        self.batchRangeLength = None
        self.global_step = 0

    def setMappings(self, mappings, embeddings):
        self.embeddings = embeddings
        self.mappings = mappings

    def setBatchRangeLength(self, batchRangeLength):
        self.batchRangeLength = batchRangeLength

    def setDataset(self, datasets, data, mainModelName=None):
        self.datasets = datasets
        self.data = data
        #print("Set dataset : {}".format(self.data['ATIS_DEBUG']['trainMatrix']))
        # Create some helping variables
        self.mainModelName = mainModelName
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
        self.modelNames = sorted(list(self.datasets.keys()))
        self.evaluateModelNames = []
        self.labelKeys = {}
        self.idx2Labels = {}
        self.trainMiniBatchRanges = None
        self.trainSentenceLengthRanges = None

        # sort model names
        # jadiin yang main pertama
        if  self.mainModelName is not None:
            print("Main model is : {}, Current list of model name is {}".format(self.mainModelName, self.modelNames))
            if self.modelNames.index(self.mainModelName) != 0:
                swap_idx = self.modelNames.index(self.mainModelName)
                temp = self.modelNames[0]
                self.modelNames[0] = self.modelNames[swap_idx]
                self.modelNames[swap_idx] = temp

                print("List of model names is changed : {}".format(self.modelNames))

        for modelName in self.modelNames:
            labelKey = self.datasets[modelName]['label']
            #print("Label here : {}".format(self.datasets[modelName]['label']))
            #print("Keys : {}".format(self.mappings[labelKey]))
            self.labelKeys[modelName] = labelKey
            self.idx2Labels[modelName] = {v: k for k, v in self.mappings[labelKey].items()}
            
            if self.datasets[modelName]['evaluate']:
                self.evaluateModelNames.append(modelName)
            
            logging.info("--- %s ---" % modelName)
            logging.info("%d train sentences" % len(self.data[modelName]['trainMatrix']))
            logging.info("%d dev sentences" % len(self.data[modelName]['devMatrix']))
            logging.info("%d test sentences" % len(self.data[modelName]['testMatrix']))
            
        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]
             
        self.casing2Idx = self.mappings['casing']

        
    def buildModel(self):
        print("Building the model")
        self.models = {}

        tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], trainable=False, name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue

            feature_input = Input(shape=(None,), dtype='int32', name=featureName+'_input')
            feature_embedding = Embedding(input_dim=len(self.mappings[featureName]), output_dim=self.params['addFeatureDimensions'], name=featureName+'_emebddings')(feature_input)

            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        

        # :: Character Embeddings ::
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            logging.info("Pad words to uniform length for characters embeddings")
            all_sentences = []
            for dataset in self.data.values():
                for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:
                    for sentence in data:
                        all_sentences.append(sentence)

            self.padCharacters(all_sentences)
            logging.info("Words padded to %d characters" % (self.maxCharLen))
            
            charset = self.mappings['characters']
            charEmbeddingsSize = self.params['charEmbeddingsSize']
            maxCharLen = self.maxCharLen
            charEmbeddings= []
            for _ in charset:
                limit = math.sqrt(3.0/charEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, charEmbeddingsSize) 
                charEmbeddings.append(vector)
                
            charEmbeddings[0] = np.zeros(charEmbeddingsSize) #Zero padding
            charEmbeddings = np.asarray(charEmbeddings)
            
            chars_input = Input(shape=(None,maxCharLen), dtype='int32', name='char_input')
            chars = TimeDistributed(Embedding(input_dim=charEmbeddings.shape[0], output_dim=charEmbeddings.shape[1],  weights=[charEmbeddings], trainable=True, mask_zero=True), name='char_emd')(chars_input)
            
            if self.params['charEmbeddings'].lower() == 'lstm': #Use LSTM for char embeddings from Lample et al., 2016
                charLSTMSize = self.params['charLSTMSize']
                chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, return_sequences=False)), name="char_lstm")(chars)
            else: #Use CNNs for character embeddings from Ma and Hovy, 2016
                charFilterSize = self.params['charFilterSize']
                charFilterLength = self.params['charFilterLength']
                chars = TimeDistributed(Conv1D(charFilterSize, charFilterLength, padding='same'), name="char_cnn")(chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)
            
            mergeInputLayers.append(chars)
            inputNodes.append(chars_input)
            self.params['featureNames'].append('characters')
            
        # :: Task Identifier :: 
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()
            
            taskID_input = Input(shape=(None,), dtype='int32', name='task_id_input')
            taskIDMatrix = np.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = Embedding(input_dim=taskIDMatrix.shape[0], output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix], trainable=False, name='task_id_embedding')(taskID_input)
        
            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.params['featureNames'].append('taskID')

        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]
        
        
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:      
            if isinstance(self.params['dropout'], (list, tuple)):
                print("LSTM Size loop 1 {}".format(size))
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                """ Naive dropout """
                print("LSTM Size loop 1 {}".format(size))
                shared_layer = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer) 
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            
            cnt += 1
            
            
        for modelName in self.modelNames:
            output = shared_layer
            
            modelClassifier = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']

            if not isinstance(modelClassifier, (tuple, list)):
                modelClassifier = [modelClassifier]
            
            cnt = 1
            for classifier in modelClassifier:
                n_class_labels = len(self.mappings[self.labelKeys[modelName]])

                if classifier == 'Softmax':
                    output = TimeDistributed(Dense(n_class_labels, activation='softmax'), name=modelName+'_softmax')(output)
                    lossFct = 'sparse_categorical_crossentropy'
                elif classifier == 'CRF':
                    output = TimeDistributed(Dense(n_class_labels, activation=None),
                                             name=modelName + '_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':
                            
                    size = classifier[1]
                    if isinstance(self.params['dropout'], (list, tuple)): 
                        output = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """ 
                        output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output) 
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)                    
                else:
                    assert(False) #Wrong classifier
                    
                cnt += 1
                
            # :: Parameters for the optimizer ::
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']
            
            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']
            
            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop': 
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=0.1, **optimizerParams)
            
            
            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)
            
            model.summary(line_length=200)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))

            self.models[modelName] = model
        


    def trainModel(self):
        #print("Beginning of trainModel  :  {}".format(self.data['ATIS_DEBUG']['trainMatrix']))
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:            
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch]) 


        total_sentence = defaultdict(int)
        #print("Before custom minibatch iterate :  {}".format(self.data['ATIS_DEBUG']['trainMatrix']))
        '''
        for batch in self.custom_minibatch_iterate_dataset():
            print("CUSTOM BATCH : {}".format(batch))

        for batch in self.minibatch_iterate_dataset() :
            print("DEFAULT BATCH : {}".format(batch))
        os._exit(2)
        '''
        '''
        for batch in self.custom_minibatch_iterate_dataset() :
            for idx, modelName in enumerate(self.modelNames):
                if modelName in batch :
                    nnLabels = batch[modelName][0]
                    nnInput = batch[modelName][1:]
                    #print("CUSTOM LOHAAA LEN NN INPUT {} {} {} {} {} {} <<".format(modelName, len(nnInput), type(nnInput), len(nnInput[0]), nnInput[0],  len(nnInput[1]), nnInput[1], len(nnInput[2]), nnInput[2]))
                    total_sentence[modelName] += len(nnInput[0])
                    self.models[modelName].train_on_batch(nnInput, nnLabels)
        '''



        for batch in self.minibatch_iterate_dataset():
            self.global_step += 1
            for idx, modelName in enumerate(self.modelNames):
                if modelName in batch :
                    #print("Training : {}".format(modelName))
                    nnLabels = batch[modelName][0]
                    nnInput = batch[modelName][1:]
                    #print("DEFAULT LOHAAA LEN NN INPUT {} {} {} {} {} {} <<".format(modelName, len(nnInput), type(nnInput), len(nnInput[0]), nnInput[0],  len(nnInput[1]), nnInput[1], len(nnInput[2]), nnInput[2]))
                    total_sentence[modelName] += len(nnInput[0])
                    self.models[modelName].train_on_batch(nnInput, nnLabels)

        print("Total training sentence : {}".format(total_sentence))

    def custom_minibatch_iterate_dataset(self, modelNames = None):

        for modelName in self.modelNames :
            trainData = self.data[modelName]['trainMatrix']
            #print("Inside custom_minibatch_iterate_dataset {}".format(trainData))
            idxs = [ idx for idx in range(len(trainData))]
            random.shuffle(idxs)

            nbTotalMiniBatches = math.ceil(len(idxs) / self.params['miniBatchSize'])
            print(" Total data point for {} is {}, total minibatch is {} with minibatchsize : {}".format(modelName, len(idxs), nbTotalMiniBatches, self.params['miniBatchSize']))

            minibatch_ranges = []
            start_range_idx = 0
            for nb_minibatch in range(nbTotalMiniBatches):
                if nb_minibatch + 1 < nbTotalMiniBatches:
                    minibatch_ranges.append((start_range_idx, start_range_idx  + self.params['miniBatchSize']))
                    start_range_idx += self.params['miniBatchSize']
            minibatch_ranges.append((start_range_idx, start_range_idx + len(idxs) % self.params['miniBatchSize']))

            #print("Ranges for the model {} is {}".format(modelName, minibatch_ranges))

            batches = {}
            for data_range in minibatch_ranges:
                batches.clear()
                trainMatrix = self.data[modelName]['trainMatrix']

                '''
                DEFAULT
                trainMatrix = self.data[modelName]['trainMatrix']
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])]
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                print("DEFAULT, type of labels is {} with the shape of {}".format(type(labels), labels.shape))
                batches[modelName] = [labels]
                
                '''
                labels = np.asarray([ trainMatrix[idxs[minibatch_idx]][self.labelKeys[modelName]] for minibatch_idx in range(data_range[0], data_range[1])])
                #print("CUSTOM, type {} LABELS : {}".format(type(labels), labels))

                labels = np.expand_dims(labels, -1)
                #print("CUSTOM, type of labels is {} with the shape of {}".format(type(labels), labels.shape))
                batches[modelName] = [labels]

                for featureName in self.params['featureNames']:
                    inputData = np.asarray([trainMatrix[idxs[minibatch_idx]][featureName] for minibatch_idx in range(data_range[0], data_range[1])])
                    #print("CUSTOM, type of inputData is {} with the shape of {}".format(type(inputData), inputData.shape))
                    batches[modelName].append(inputData)
                #print("CUSTOM BATCHES : {}".format(batches))
                yield(batches)


    def minibatch_iterate_dataset(self, modelNames = None):
        """ Create based on sentence length mini-batches with approx. the same size. Sentences and 
        mini-batch chunks are shuffled and used to the train the model """
        
        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            self.trainSentenceLengthRanges = {}
            self.trainMiniBatchRanges = {}            
            for modelName in self.modelNames:
                trainData = self.data[modelName]['trainMatrix']
                print("Number of training data for {} is {}".format(modelName, len(trainData)))
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by sentence length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))

                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                      
                self.trainSentenceLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges



        if modelNames == None:
            modelNames = self.modelNames



        #Shuffle training data
        for modelName in modelNames:      
            #1. Shuffle sentences that have the same length
            x = self.data[modelName]['trainMatrix']
            for dataRange in self.trainSentenceLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    random.seed(1337)
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
               
            #2. Shuffle the order of the mini batch ranges
            random.seed(1337)
            random.shuffle(self.trainMiniBatchRanges[modelName])
     
        
        #Iterate over the mini batch ranges
        print("Initial range Batches for the models ")

        if self.mainModelName != None:
            rangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
        else:
            rangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])

        #print("Self.mainModelName : {}  rangeLength : {} max : {}".format(self.mainModelName, rangeLength, maxRangeLength))

        # Same number of batches between main task and aux tasks
        if self.batchRangeLength is not None and self.batchRangeLength == "max" :
            print("Picking up max range length")
            from copy import deepcopy
            maxRangeLength = max([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])
            rangeLength = maxRangeLength

            for modelName in modelNames :
                # Check how many sentences in the range
                total_sent = 0
                for elmt in self.trainMiniBatchRanges[modelName]:
                    total_sent += elmt[1] - elmt[0]
                #print("Total sent for {} is {}".format(modelName, total_sent))
                # compute the range difference for this model batch to the max
                mult = maxRangeLength / len(self.trainMiniBatchRanges[modelName])
                modulo = maxRangeLength % len(self.trainMiniBatchRanges[modelName])
                self.trainMiniBatchRanges[modelName] = self.trainMiniBatchRanges[modelName] * int(mult)
                #print("Len of self.trainMiniBatchRanges[modelName] {} ".format(len(self.trainMiniBatchRanges[modelName])))
                temp = deepcopy(self.trainMiniBatchRanges[modelName])

                for i in range(modulo) :
                    #print("{} {}".format(i, len(temp)))
                    self.trainMiniBatchRanges[modelName].append(temp[i % len(temp)])


        print("Final range Batches for the models ")
        for modelName in modelNames:
            print("Range length for {} is {}".format(modelName, len(self.trainMiniBatchRanges[modelName])))
            total_sent = 0

            for data_range in self.trainSentenceLengthRanges[modelName]:
                total_sent += data_range[1] - data_range[0]
                #if modelName == "ATIS" :
                #    print("{} {}".format(modelName, data_range))
            print("Total sent for {} is {}".format(modelName, total_sent))

        batches = {}
        counter  = { modelName : 0  for modelName in modelNames}

        for idx in range(rangeLength):
            batches.clear()
            for modelName in modelNames:
                if self.batchRangeLength == "max" and len(self.trainMiniBatchRanges[modelName]) < rangeLength :
                    if idx % (math.floor(rangeLength / len(self.trainMiniBatchRanges[modelName]))) == 0 :
                        if counter[modelName] < len(self.trainMiniBatchRanges[modelName]):
                            trainMatrix = self.data[modelName]['trainMatrix']
                            dataRange = self.trainMiniBatchRanges[modelName][counter[modelName]]
                            cnt = 0
                            #print("dataRange : {}".format(dataRange))
                            labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])

                            labels = np.expand_dims(labels, -1)

                            batches[modelName] = [labels]


                            for featureName in self.params['featureNames']:
                                inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                                batches[modelName].append(inputData)
                            counter[modelName] += 1
                            #print("Counter for {} : {} LEN LABELS : :{}".format(modelName, counter[modelName], len(labels)))
                else :
                    trainMatrix = self.data[modelName]['trainMatrix']
                    dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])]
                    labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                    if modelName != self.mainModelName and self.batchRangeLength == "same":
                        labels = labels[:len(batches[self.mainModelName][0])]
                        #print("Len of main model is {}".format(len(batches[self.mainModelName][0])))
                    #print("DEFAULT, type {} LABELS : {}".format(type(labels), labels))

                    labels = np.expand_dims(labels, -1)
                    #print("DEFAULT, type of labels is {} with the shape of {}".format(type(labels), labels.shape))
                    batches[modelName] = [labels]

                    cnt_feature = 0
                    for featureName in self.params['featureNames']:
                        inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                        if modelName != self.mainModelName and self.batchRangeLength == "same":
                            # labels = labels[:len(batches[self.mainModelName][0])]
                            inputData = inputData[:len(batches[self.mainModelName][cnt_feature])]
                            #print("Len of main model input data is {}  featureName : {}".format(len(batches['ATIS'][1]), featureName))
                            #print("Len of main model input data is {} featureName : {}".format(len(batches['ATIS'][2]), featureName))
                        #print("DEFAULT, type of inputData is {} with the shape of {}".format(type(inputData),inputData.shape))
                        batches[modelName].append(inputData)
                        cnt_feature += 1
                    counter[modelName] += 1
            #print(" DEFAULT BATCHES : {}".format(batches))
            yield batches   
            
        print(counter)
        
    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsSavePath = open(resultsFilepath, 'w')
        else:
            self.resultsSavePath = None
        
    def fit(self, epochs):
        #print("Inside FIT : {}".format(self.data['ATIS_DEBUG']['trainMatrix']))
        if self.models is None:
            self.buildModel()
            #print("After Build Model : {}".format(self.data['ATIS_DEBUG']['trainMatrix']))

        total_train_time = 0
        max_dev_score = {modelName: -1 for modelName in self.models.keys()} # DANGER
        max_test_score = {modelName: -1 for modelName in self.models.keys()}
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()
            print("Epoch {}".format((epoch + 1)))
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time()

            # Training Part
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            

            # Evaluation Part
            start_time = time.time() 
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
                dev_score, test_score = self.computeScore(modelName, self.data[modelName]['devMatrix'], self.data[modelName]['testMatrix'],epoch=epoch)

                if dev_score > max_dev_score[modelName]:
                    self.final_max_dev_score = dev_score
                    max_dev_score[modelName] = dev_score
                    max_test_score[modelName] = test_score
                    no_improvement_since = 0

                    #Save the model
                    if self.modelSavePath != None:
                        print("IMPROVEMENT... Saving the model {}".format(datetime.datetime.now().time()))
                        self.saveModel(modelName, epoch, dev_score, test_score)

                    labelKey = self.labelKeys[modelName]
                    idx2Label = self.idx2Labels[modelName]

                    dev_sentences = self.data[modelName]['devMatrix']
                    test_sentences = self.data[modelName]['testMatrix']

                    correctDevLabels  =  [dev_sentences[idx][labelKey] for idx in range(len(dev_sentences))]
                    correctTestLabels = [test_sentences[idx][labelKey] for idx in range(len(test_sentences))]

                    if self.predictionSavePath != None:
                        print("IMPROVEMENT... Saving the prediction {}".format(datetime.datetime.now().time()))
                        self.savePredictionResults(modelName, dev_sentences, correctDevLabels,   self.current_dev_prediction, idx2Label, epoch, "dev")
                        self.savePredictionResults(modelName, test_sentences, correctTestLabels, self.current_test_prediction, idx2Label, epoch, "test")

                else:
                    no_improvement_since += 1
                    
                    
                if self.resultsSavePath != None:
                    self.resultsSavePath.write("\t".join(map(str, [epoch + 1, modelName, dev_score, test_score, max_dev_score[modelName], max_test_score[modelName]])))
                    self.resultsSavePath.write("\n")
                    self.resultsSavePath.flush()

                print("Max: {:.4f} dev; {:.4f} test".format(max_dev_score[modelName], max_test_score[modelName]))
                logging.info("Max: %.4f dev; %.4f test" % (max_dev_score[modelName], max_test_score[modelName]))
                logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break

        return max_dev_score, max_test_score

            
    def tagSentences(self, sentences):
        # Pad characters
        if 'characters' in self.params['featureNames']:
            self.padCharacters(sentences)

        labels = {}
        for modelName, model in self.models.items():
            paddedPredLabels = self.predictLabels(model, sentences)
            predLabels = []
            for idx in range(len(sentences)):
                unpaddedPredLabels = []
                for tokenIdx in range(len(sentences[idx]['tokens'])):
                    if sentences[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

                predLabels.append(unpaddedPredLabels)

            idx2Label = self.idx2Labels[modelName]
            labels[modelName] = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]

        return labels
            
    
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths

    def predictLabels(self, model, sentences):
        predLabels = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences)
        
        for indices in sentenceLengths.values():
            #indices itu adalah idx sentence dengan panjang tertentu
            nnInput = []
            raw_text = []
            for featureName in self.params['featureNames']:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                nnInput.append(inputData)
            #print(indices)
            raw_texts = [sentences[idx]['raw_tokens'] for idx in indices]
            #print("Len NN Input : {}".format(len(nnInput)))
            #print("NN input content : {}".format(nnInput))
            #print("Len Raw Text : {} Raw Text : {}".format(len(raw_texts), raw_texts))


            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) #Predict classes            
           
            #print("Idx 2 Labels {} {}".format(type(self.idx2Labels), self.idx2Labels))
            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx][:len(sentences[idx]['raw_tokens'])]
                #print("Pred Labels idx {}".format(predLabels[idx]))
                #print("Prediction type : {}".format(type(predLabels[idx])))
                #print("Prediction : {}".format(predLabels[idx]))
                predIdx += 1
            #exit()
        return predLabels
    
   
    def computeScore(self, modelName, devMatrix, testMatrix, epoch=0):
        if self.labelKeys[modelName].endswith('_BIO') or self.labelKeys[modelName].endswith('_IOBES') or self.labelKeys[modelName].endswith('_IOB'):
            return self.computeF1Scores(modelName, devMatrix, testMatrix, epoch = epoch)
        else:
            return self.computeAccScores(modelName, devMatrix, testMatrix)   

    def computeF1Scores(self, modelName, devMatrix, testMatrix, epoch = 0):
        #train_pre, train_rec, train_f1 = self.computeF1(modelName, self.datasets[modelName]['trainMatrix'])
        #print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (train_pre, train_rec, train_f1)
        
        dev_pre, dev_rec, dev_f1 = self.computeF1(modelName, devMatrix, mode = 'dev', epoch = epoch)
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.2f" % (dev_pre, dev_rec, dev_f1))

        test_pre, test_rec, test_f1 = self.computeF1(modelName, testMatrix, mode = 'test', epoch = epoch)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.2f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1
    
    def computeAccScores(self, modelName, devMatrix, testMatrix):
        dev_acc = self.computeAcc(modelName, devMatrix)
        test_acc = self.computeAcc(modelName, testMatrix)
        
        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        
        return dev_acc, test_acc   
        
        
    def computeF1(self, modelName, sentences, mode="", epoch = 0):
        labelKey = self.labelKeys[modelName]
        print("Label key : {}".format(labelKey))
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]
        
        correctLabels = [sentences[idx][labelKey][:len(sentences[idx]['raw_tokens'])] for idx in range(len(sentences))]
        #print("Correct labels : {}".format(correctLabels))
        #os._exit(2)
        predLabels = self.predictLabels(model, sentences)
        #print("Prediction labels : {}".format(predLabels))


        if mode == "dev" :
            self.current_dev_prediction = predLabels
        if mode == "test" :
            self.current_test_prediction = predLabels

        labelKey = self.labelKeys[modelName]
        encodingScheme = labelKey[labelKey.index('_')+1:]
        
        #pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'O', encodingScheme)
        #pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'B', encodingScheme)
        pre, rec, f1 = BIOF1Validation.compute_f1_conll(correctLabels, predLabels, idx2Label)
        #logging.info("ConLL version p : {:.2f} r: {:.2f} f1: {:.2f}".format(pre_conll, rec_conll, f1_conll))
        #if f1_b > f1:
        #    logging.info("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
        #    pre, rec, f1 = pre_b, rec_b, f1_b
        
        return pre, rec, f1
    
    def computeAcc(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        predLabels = self.predictLabels(self.models[modelName], sentences) 
        
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1

  
        return numCorrLabels/float(numLabels)
    
    def padCharacters(self, sentences):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = self.params['maxCharLength']
        if maxCharLen <= 0:
            for sentence in sentences:
                for token in sentence['characters']:
                    maxCharLen = max(maxCharLen, len(token))
          

        for sentenceIdx in range(len(sentences)):
            for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
                token = sentences[sentenceIdx]['characters'][tokenIdx]

                if len(token) < maxCharLen: #Token shorter than maxCharLen -> pad token
                    sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')
                else: #Token longer than maxCharLen -> truncate token
                    sentences[sentenceIdx]['characters'][tokenIdx] = token[0:maxCharLen]
    
        self.maxCharLen = maxCharLen
        
    def addTaskIdentifier(self):
        """ Adds an identifier to every token, which identifies the task the token stems from """
        taskID = 0
        for modelName in self.modelNames:
            dataset = self.data[modelName]
            for dataName in ['trainMatrix', 'devMatrix', 'testMatrix']:            
                for sentenceIdx in range(len(dataset[dataName])):
                    dataset[dataName][sentenceIdx]['taskID'] = [taskID] * len(dataset[dataName][sentenceIdx]['tokens'])
            
            taskID += 1


    def saveModel(self, modelName, epoch, dev_score, test_score):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')

        savePath = self.modelSavePath.replace("[ModelName]", modelName)

        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Model "+savePath+" already exists. Model will be overwritten")

        self.models[modelName].save(savePath, True)

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['modelName'] = modelName
            h5file.attrs['labelKey'] = self.datasets[modelName]['label']

    def saveParams(self, paramFilePath):

        with open(paramFilePath, "w") as f:
            f.write(json.dumps(self.params))

    def saveParamTuningResults(self, tuningFilePath):
        mode = ""
        if os.path.exists(tuningFilePath) :
            mode = "a"
        else :
            mode = "w"
        print(self.params['dropout'])
        print(self.final_max_dev_score)
        with open(tuningFilePath, mode) as f:
            f.write("{} {} {}\n".format(self.params['dropout'][0], self.params['dropout'][1],self.final_max_dev_score))

    def savePredictionResults(self, modelName, sentences, correctLabels,predLabels, idx2Label,epoch, mode = ""):
        if self.predictionSavePath== None:
            raise ValueError('predictionSavePath not specified.')
        #savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace( "[Epoch]", str(epoch + 1)).replace("[ModelName]", modelName)
        #print(type(self.predictionSavePath))
        #savePath = self.predictionSavePath.replace("[Epoch]", str(epoch + 1))#.replace("[ModelName]", modelName).replace("[Data]", mode)

        savePath = self.predictionSavePath.replace("[ModelName]", modelName).replace("[Data]", mode)
        #print(savePath)
        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Prediction " + savePath + " already exists. Prediction will be overwritten")

        with open(savePath, "w") as f:
            #print("Saving to {}".format(self.predictionSavePath))
            for idx in range(len(sentences)):
                #print("Sentence :       {}".format(sentences[idx]['raw_tokens']))
                #print("Predicted tags : {}{}".format(type(predLabels[idx]), predLabels[idx]))
                correctLabel = [idx2Label[labelId] for labelId in correctLabels[idx]]
                #print("Correct label : {}".format(correctLabel))
                predictedLabel = [idx2Label[labelId] for labelId in predLabels[idx]]

                #print("Label  : {}".format(predictedLabel))
                for token_id in range(len(sentences[idx]['raw_tokens'])):
                    f.write(sentences[idx]['raw_tokens'][token_id]+" "+correctLabel[token_id]+" "+predictedLabel[token_id]+"\n")
                f.write("\n")

    def get_params(self):
        return self.params
    @staticmethod
    def loadModel(modelPath):
        import h5py
        import json
        from .keraslayers.ChainCRF import create_custom_objects

        model = keras.models.load_model(modelPath, custom_objects=create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']

        bilstm = BiLSTM(params)
        bilstm.setMappings(mappings, None)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm