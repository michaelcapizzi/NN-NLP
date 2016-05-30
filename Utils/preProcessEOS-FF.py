import PreProcessing as pre
from gensim import models as g
import sys
import pickle

#script to solely convert EOS plain text data to vector representations to be used in FF models

#sys.argv[1] = path to training file (from training_instances)
#sys.argv[2] = path to testing file
#sys.argv[3] = W2V vectors to use
#sys.argv[4] = number of lines to process
#sys.argv[5] = lemmatize? (boolean)
#sys.argv[6] = location for pickled training data
#sys.argv[7] = location for pickled testing data



print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

trainPath = sys.argv[1]
testPath = sys.argv[2]
num_lines = int(sys.argv[4])
lemmatize = bool(sys.argv[5])
w2vDimension = len(w2v["the"])
pickleTrain = sys.argv[6]
pickleTest = sys.argv[7]

print("processing training")
f = open(trainPath, "rb")
#start the server
processor = pre.initializeProcessor()
#starting processors server
pre.startServer(processor)
#initialize vector to house training and testing instances
training_vectors = []
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
            [training_vectors.append(t) for t in tokensVectorLabels]
        except Exception as ex:
            print("ERROR in annotating.  Skipping line.")
f.close()

print("separating training data in half")
l = len(training_vectors)
train_1 = training_vectors[0:l/2]
train_2 = training_vectors[l/2:l+1]

print("pickling training data into two parts")
fTrain1 = open("training_instances/01-" + pickleTrain, "wb")
pickle.dump(train_1, fTrain1)
fTrain1.close()
fTrain2 = open("training_instances/02-" + pickleTrain, "wb")
pickle.dump(train_2, fTrain2)
fTrain2.close()

print("processing testing")
testing_vectors = []
#open file
f = open(testPath, "rb")
#counter to keep track of the number of lines to process
c = 0
#iterate through each line
for line in f:
    if c <= num_lines or num_lines == 0:
        #set counter for total number of lines
        c+=1
        #process line
        tokensLabels = pre.convertLineForEOS(line, processor, lemmatize)
        #unpack tokens and labels
        tokens, labels = zip(*tokensLabels)
        #convert tokens to vector representation
        tokensVector = pre.convertSentenceToVec(tokens, w2v, w2vDimension)
        tokensVectorLabels = zip(tokensVector, labels)
        [testing_vectors.append(t) for t in tokensVectorLabels]
f.close()

print("pickling testing data")
fTest = open(pickleTest, "wb")
pickle.dump(testing_vectors, fTest)
fTest.close()

print("stopping server")
processor.stop_server()