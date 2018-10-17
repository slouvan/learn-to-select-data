from __future__ import print_function
import logging
"""
Computes the F1 score on BIO tagged data

@author: Nils Reimers
"""



def compute_f1_token_basis(predictions, correct, O_Label): 
       
    prec = compute_precision_token_basis(predictions, correct, O_Label)
    rec = compute_precision_token_basis(correct, predictions, O_Label)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        for idx in range(len(guessed)):
            
            if guessed[idx] != O_Label:
                count += 1
               
                if guessed[idx] == correct[idx]:
                    correctCount += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision


def compute_f1(predictions, correct, idx2Label, correctBIOErrors='No', encodingScheme='BIO'): 
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
            
    encodingScheme = encodingScheme.upper()
    
    
    if encodingScheme == 'IOBES':
        convertIOBEStoBIO(label_pred)
        convertIOBEStoBIO(label_correct)                 
    elif encodingScheme == 'IOB':
        convertIOBtoBIO(label_pred)
        convertIOBtoBIO(label_correct)
            
                    
    
    #print("Correct BIO Errors : {}".format(correctBIOErrors))
    #checkBIOEncoding(label_pred, correctBIOErrors)

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)
    
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
        
    return prec, rec, f1

def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd

def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType

def do_computeF1ScoreCONLL(correct_slots, pred_slots):

    print("Evaluating F1 score of {} number of predictions ".format(len(correct_slots)))
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                    __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                    (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1
    if foundPredCnt > 0:
        precision = 100.0 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100.0 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0

    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

def compute_f1_conll(correct, predictions, idx2Label):
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    precision, recall, f1 = do_computeF1ScoreCONLL(label_correct, label_pred)

    return precision, recall, f1
def convertIOBtoBIO(dataset):
    """ Convert inplace IOB encoding to BIO encoding """
    for sentence in dataset:
        prevVal = 'O'
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'I':
                if prevVal == 'O' or prevVal[1:] != sentence[pos][1:]:
                    sentence[pos] = 'B'+sentence[pos][1:] #Change to begin tag

            prevVal = sentence[pos]

def convertIOBEStoBIO(dataset):
    """ Convert inplace IOBES encoding to BIO encoding """    
    for sentence in dataset:
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'S':
                sentence[pos] = 'B'+sentence[pos][1:]
            elif firstChar == 'E':
                sentence[pos] = 'I'+sentence[pos][1:]
                
def testEncodings():
    """ Tests BIO, IOB and IOBES encoding """
    
    goldBIO   = [['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'B-PER', 'I-PER'], ['O', 'B-PER', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'I-PER'], ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'B-PER']]
    
    
    print("--Test IOBES--")
    goldIOBES = [['O', 'B-PER', 'E-PER', 'O', 'S-PER', 'B-PER', 'E-PER'], ['O', 'S-PER', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'I-PER', 'E-PER'], ['B-LOC', 'I-LOC', 'E-LOC', 'S-PER', 'B-PER', 'I-PER', 'E-PER', 'O', 'S-LOC', 'S-PER']]
    convertIOBEStoBIO(goldIOBES)
    
    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert(goldBIO[sentenceIdx][tokenIdx] == goldIOBES[sentenceIdx][tokenIdx])
            
    print("--Test IOB--")        
    goldIOB   = [['O', 'I-PER', 'I-PER', 'O', 'I-PER', 'B-PER', 'I-PER'], ['O', 'I-PER', 'I-LOC', 'I-LOC', 'O', 'I-PER', 'I-PER', 'I-PER'], ['I-LOC', 'I-LOC', 'I-LOC', 'I-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'I-LOC', 'I-PER']]
    convertIOBtoBIO(goldIOB)
    
    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert(goldBIO[sentenceIdx][tokenIdx] == goldIOB[sentenceIdx][tokenIdx])
            
    print("test encodings completed")
    


def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        
        
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': #A new chunk starts
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision

def checkBIOEncoding(predictions, correctBIOErrors):
    errors = 0
    labels = 0
    
    for sentenceIdx in range(len(predictions)):
        labelStarted = False
        labelClass = None
        

        for labelIdx in range(len(predictions[sentenceIdx])): 
            label = predictions[sentenceIdx][labelIdx]      
            if label.startswith('B-'):
                labels += 1
                labelStarted = True
                labelClass = label[2:]
            
            elif label == 'O':
                labelStarted = False
                labelClass = None
            elif label.startswith('I-'):
                if not labelStarted or label[2:] != labelClass:
                    errors += 1        
                    
                    if correctBIOErrors.upper() == 'B':
                        predictions[sentenceIdx][labelIdx] = 'B-'+label[2:]
                        labelStarted = True
                        labelClass = label[2:]
                    elif correctBIOErrors.upper() == 'O':
                        predictions[sentenceIdx][labelIdx] = 'O'
                        labelStarted = False
                        labelClass = None
            else:
                assert(False) #Should never be reached
           
    
    if errors > 0:
        labels += errors
        logging.info("Wrong BIO-Encoding %d/%d labels, %.2f%%" % (errors, labels, errors/float(labels)*100),)


def compute_f1_argument(predictions, correct, idx2Label):     
    prec = compute_argument_chunk_precision(predictions, correct)
    rec = compute_argument_chunk_precision(correct, predictions)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_f1_argument_token_basis(predictions, correct, idx2Label):     
    prec = compute_argument_token_precision(predictions, correct)
    rec = compute_argument_token_precision(correct, predictions)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_argument_token_precision(predictions, correct):
    count = 0
    correctCount = 0
    
    for sentenceIdx in range(len(predictions)):
        for tokenIdx in range(len(predictions[sentenceIdx])):
            for argIdx in range(len(predictions[sentenceIdx][tokenIdx])):
                pred = predictions[sentenceIdx][tokenIdx][argIdx]
                corr = correct[sentenceIdx][tokenIdx][argIdx]
                
                if pred:
                    count += 1
                    
                    if pred == corr:
                        correctCount += 1
    
    if count == 0:
        return 0
    
    return correctCount / float(count)

def compute_argument_chunk_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        assert(len(guessed_sentences[sentenceIdx]) == len(correct_sentences[sentenceIdx]))
        
        for argIdx in range(len(guessed_sentences[sentenceIdx][0])):
            idx = 0
            guessed = guessed_sentences[sentenceIdx]
            correct = correct_sentences[sentenceIdx]
            while idx < len(guessed):
                if guessed[idx][argIdx]: #A new chunk starts
                    count += 1
                    
                    if guessed[idx][argIdx] == correct[idx][argIdx]:
                        idx += 1
                        correctlyFound = True
                        
                        while idx < len(guessed) and guessed[idx][argIdx]: #Scan until it no longer starts with I
                            if guessed[idx][argIdx] != correct[idx][argIdx]:
                                correctlyFound = False
                            
                            idx += 1
                        
                        if idx < len(guessed):
                            if correct[idx][argIdx]: #The chunk in correct was longer
                                correctlyFound = False
                            
                        
                        if correctlyFound:
                            correctCount += 1
                    else:
                        idx += 1
                else:  
                    idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision


if __name__ == "__main__":
    testEncodings()