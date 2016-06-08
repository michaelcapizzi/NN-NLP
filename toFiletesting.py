import Utils.PreProcessing as pre
from gensim import models as g
import numpy as np




print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format("w2v_Goldberg.txt.gz", binary=False)

inPath = "training_instances/segmenter_test.master"
vectorPath = "/home/mcapizzi/Desktop/vectors.csv"
labelPath = "/home/mcapizzi/Desktop/labels.csv"

f = open(inPath, "rb")
#start the server
processor = pre.initializeProcessor()
#starting processors server
pre.startServer(processor)
#all lines
lines = f.readlines()
f.close()

fVector = open(vectorPath, "a")
fLabel = open(labelPath, "a")

for line in lines[0:10]:
    tokensLabels = pre.convertLineForEOS(line, processor, True)
    # #process line
    # #unpack tokens and labels
    tokens, labels = zip(*tokensLabels)
    # #convert tokens to vector representation
    tokensVector = pre.convertSentenceToVec(tokens, w2v, len(w2v["the"]))
    tokensVectorLabels = zip(tokensVector, labels)
    #write to file
    for token, label in tokensVectorLabels:
        np.savetxt(fVector, token.reshape((1,300)), delimiter=",", newline="\n")
        np.savetxt(fLabel, np.asarray(label).reshape((1,1)), newline="\n")

fVector.close()
fLabel.close()

