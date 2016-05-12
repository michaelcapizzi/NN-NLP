from processors import *
import numpy as np
import re
import itertools


"""
Currently not using preprocessing methods available in keras.  Using pyprocessors [https://github.com/myedibleenso/py-processors]
"""

#initialize a processor
def initializeProcessor():
    return Processor(port=8886)


#start server
def startServer(proc):
    proc.start_server(jar_path="processors-server.jar")


#annotate text
def annotate(proc, text):
    return proc.annotate(text)


"""
other methods
"""

#sort sequences by length
def sortBySeqLength(seqs):
    #instantiate empty dictionary
    dict = {}

    for s in seqs:
        #get length
        length = len(s)
        #if length alreayd in keys
        if length in dict.keys():
            #append this seq to list
            dict[length].append(s)
        else:
            #add key and this sequence as first item
            dict[length] = [s]

    return dict


#pad all sentences to length of longest sentence
def padToLongest(dictOfSeqs, w2vDim):
    #get lengths from dictionary
    keys = dictOfSeqs.keys()
    #get longest length
    longest = max(keys)
    #iterate through dictionary keys and pad
    new_dict = {}
    for k in keys:
        if k != longest:
            seqs = dictOfSeqs[k]
            padded_seqs = []
            for s in seqs:
                #pad with np.zeros(w2vDim) at the end til it's the same length as longest sequence
                for i in range(longest-k):
                    s.append(np.zeros(w2vDim))
                #add to padded_seqs
                padded_seqs.append(s)
            #add to new dictionary
            new_dict[k] = padded_seqs
        else:
            new_dict[k] = dictOfSeqs[k]

    return new_dict


#pad single sentence (of word vectors) to arbitrary length
def padEmbeddingToConstant(sentence, w2vDim, maxLength):
    #pad sentence
    for i in range(maxLength-len(sentence)):
        sentence.append(np.zeros(w2vDim))

    return sentence


#pad single sentence (of one-hot vectors) to arbitrary length
def padOneHotToConstant(sentence, maxLength):
    #pad sentence
    for i in range(maxLength-len(sentence)):
        sentence.append(0)

    return sentence


#get word vector
def getVector(word, gensimModel, w2vDim):
    if word in gensimModel:
        return gensimModel[word]
    else:
        return np.zeros(w2vDim)


#convert a list of words to list of word embeddings
def convertSentenceToVec(listOfWords, gensimModel, w2vDim):
    return [getVector(word, gensimModel, w2vDim) for word in listOfWords]


#convert a list of words or lemmas to one-hot vectors
def convertSentenceToOneHots(listOfWords, wordToIDX):
    return [wordToIDX.get(word) for word in listOfWords]


#converts plain text for training/testing in EOS
# def convertForEOS(fPath, line_separated=False, lemmatize=False, p=None):
#     #open file
#     f = open(fPath, "rb")
#
#     #if lemmatizing or not line_separated, start server
#     # if lemmatize or not line_separated:
#     #     p = initializeProcessor()
#     #     startServer(p)
#
#     #variable to house final tokens and labels zipped
#     allTokensLabels = []
#
#     #regex for punctuation
#     punctRegex = r'\W'
#
#     #regex for numbers
#     numberRegex = r'\d+'
#
#     if line_separated:
#         for line in f:
#             #strip punctuation
#             clean = re.sub(punctRegex, "", line.rstrip())
#             #replace numbers
#             clean2 = re.sub(numberRegex, "number", clean.rstrip())
#             if lemmatize:
#                 #annotate
#                 sentence = annotate(p, clean2)
#                 #get lemmas
#                 tokens = sentence.lemmas
#             else:
#                 #split tokens
#                 tokens = clean2.split(" ")
#             #make labels
#             labels = [0] * len(tokens)
#             labels[-1] = 1
#             #zip
#             tokensLabels = zip(tokens, labels)
#             #add to allTokensLabels
#             [allTokensLabels.append(tl) for tl in tokensLabels]
#     else:
#         for line in f:
#             #strip punctuation
#             clean = re.sub(punctRegex, "", line.rstrip())
#             #replace numbers
#             clean2 = re.sub(numberRegex, "number", clean.rstrip())
#             #annotate
#             annotated = annotate(p, clean2)
#             for sentence in annotated.sentences:
#                 if lemmatize:
#                     tokens = sentence.lemmas
#                 else:
#                     tokens = sentence.words
#                 #make labels
#                 labels = [0] * len(tokens)
#                 labels[-1] = 1
#                 #zip
#                 tokensLabels = zip(tokens, labels)
#                 #add to allTokensLabels
#                 [allTokensLabels.append(tl) for tl in tokensLabels]
#
#     #close file
#     f.close()
#
#     #close server
#     # if lemmatize or not line_separated:
#     #     p.__del__()
#
#     return allTokensLabels

def convertLineForEOS(line, processor, lemmatize=False):
    #variable to house final tokens and labels zipped
    allTokensLabels = []

    #regex for punctuation
    punctRegex = r'[^\w\']'

    #regex for numbers
    # numberRegex = r'\d+'

    #replace numbers
    # clean = re.sub(numberRegex, "number", line.rstrip())
    clean = line.rstrip()

    #annotate
    annotated = annotate(processor, clean)

    for sentence in annotated.sentences:
        #get tokens with punctuation filtered
        if lemmatize:
            for t in range(len(sentence.lemmas)):
                token = sentence.lemmas[t]
                #if it's not a punctuation token
                #get label
                if t == len(sentence.lemmas) - 2:
                    label = 1
                else:
                    label = 0
                if not re.match(punctRegex, token) and token != "'":
                    #zip
                    tokenLabel = (token, label)
                    #add tokenLabel
                    allTokensLabels.append(tokenLabel)
        else:
            for t in range(len(sentence.words)):
                token = sentence.words[t]
                #if it's not a punctuation token
                #get label (second last token == last word)
                if t == len(sentence.lemmas) - 2:
                    label = 1
                else:
                    label = 0
                if not re.match(punctRegex, token) and token != "'":
                    #zip
                    tokenLabel = (token, label)
                    #add tokenLabel
                    allTokensLabels.append(tokenLabel)

    return allTokensLabels


