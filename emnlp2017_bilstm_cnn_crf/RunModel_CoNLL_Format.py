#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging
import argparse

parser = argparse.ArgumentParser(description="Experiment Slot Filling")
parser.add_argument("-m", "--model",  dest="model_path", help="Model Path", required=True, type=str)
parser.add_argument("-i", "--input",  dest="input_file", help="Input",   required = True, type=str)
parser.add_argument("-o", "--output", dest="output_file",help="Output", required=True,   type=str)

args = parser.parse_args()

#if len(sys.argv) < 4:
#    print("Usage: python RunModel.py modelPath inputPathToConllFile outputPathToConllFile")
#    exit()

#modelPath = sys.argv[1]
#inputPath = sys.argv[2]
#outputPath = sys.argv[3]
inputColumns = {0: "tokens", 1 : "gold"}



# :: Prepare the input ::
sentences = readCoNLL(args.input_file, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(args.model_path)
params = lstmModel.get_params()
#print("params : {}".format(params))

dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
f = None
if args.output_file is not None :
    f = open(args.output_file, "w")

for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    golds = sentences[sentenceIdx]['gold']
    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            if tokenIdx >= len (tags[modelName][sentenceIdx]) :
                #print("Different")
                #continue
                pass
            else :
                tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

                #print("%s %s %s" % (tokens[tokenIdx], golds[tokenIdx]," ".join(tokenTags)))
                if f is not None:
                    f.write("{} {} {}\n".format(tokens[tokenIdx], golds[tokenIdx]," ".join(tokenTags)))
    #print("")
    if f is not None:
        f.write("\n")