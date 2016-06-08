import PreProcessing as pre
from gensim import models as g
import sys
import numpy as np

#script to solely convert EOS plain text data to vector representations to be used in FF models

# python Utils/preProcessEOS-FF.py training_instances/segmenter_train.master training_instances/segmenter_test.master w2v_Goldberg.txt.gz 10 lemmatize training_instances/segmenter_train.csv training_instances/segmenter_test.csv



#sys.argv[1] = path to training file
#sys.argv[2] = path to testing file
#sys.argv[3] = W2V vectors to use
#sys.argv[4] = number of lines to process
#sys.argv[5] = lemmatize? (boolean)
#sys.argv[6] = location for csv'd training data
#sys.argv[7] = location for csv'd testing data



print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

trainPath = sys.argv[1]
testPath = sys.argv[2]
num_lines = int(sys.argv[4])
lemmatize = bool(sys.argv[5])
w2vDimension = len(w2v["the"])
csvTrain = sys.argv[6]
csvTest = sys.argv[7]

print("processing training")
f = open(trainPath, "rb")
#start the server
processor = pre.initializeProcessor()
#starting processors server
pre.startServer(processor)
#open files
fTrainVectors = open("vectors-" + csvTrain, "a")
fTrainLabels = open("labels-" + csvTrain, "a")
fTestVectors = open("vectors-" + csvTest, "a")
fTestLabels = open("labels-" + csvTest, "a")

#counter to keep track of the number of lines to process
c = 0
#iterate through each line
for line in f:
    if (c <= num_lines or num_lines == 0) and len(line.split(" ")) > 1 and "@" not in line and "#" not in line:
        #set counter for total number of lines
        c+=1
        try:
            tokensLabels = pre.convertLineForEOS(line, processor, lemmatize)
            print(line.rstrip(), c)
            #process line
            #unpack tokens and labels
            tokens, labels = zip(*tokensLabels)
            #convert tokens to vector representation
            tokensVector = pre.convertSentenceToVec(tokens, w2v, w2vDimension)
            tokensVectorLabels = zip(tokensVector, labels)
            #write to file
            for token, label in tokensVectorLabels:
                np.savetxt(fTrainVectors, token.reshape((1, w2vDimension)), delimiter=",", newline="\n")
                np.savetxt(fTrainLabels, np.asarray(label).reshape((1,1)), newline="\n")
        except Exception as ex:
            print("ERROR in annotating.  Skipping line.")
f.close()
fTrainVectors.close()
fTrainLabels.close()

print("processing testing")
# testing_vectors = []
#open file
f = open(testPath, "rb")
#counter to keep track of the number of lines to process
c = 0
#iterate through each line
for line in f:
    if c <= num_lines or num_lines == 0:
        #set counter for total number of lines
        c+=1
        print(line.rstrip(), c)
        #process line
        tokensLabels = pre.convertLineForEOS(line, processor, lemmatize)
        #unpack tokens and labels
        tokens, labels = zip(*tokensLabels)
        #convert tokens to vector representation
        tokensVector = pre.convertSentenceToVec(tokens, w2v, w2vDimension)
        tokensVectorLabels = zip(tokensVector, labels)
        #write to file
        for token, label in tokensVectorLabels:
            np.savetxt(fTestVectors, token.reshape((1, w2vDimension)), delimiter=",", newline="\n")
            np.savetxt(fTestLabels, np.asarray(label).reshape((1,1)), newline="\n")
f.close()
fTestVectors.close()
fTestLabels.close()

print("stopping server")
processor.stop_server()