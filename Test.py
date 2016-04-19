from gensim import models as g
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
import pickle
from collections import Counter
from multiprocessing import Process

#TODO look here! https://github.com/fchollet/keras/issues/395


#sys.argv[1] = file to process for data
#sys.argv[2] = number of lines to take from file
#sys.argv[3] = line separated?
#sys.argv[4] = word2vec file
#sys.argv[5] = max length of sentence to consider (currently using 30)
#sys.argv[6] = [optional] full path of pickled training vectors
#sys.argv[7] = [optional] full path of pickled testing vectors


#get embeddings
print("loading embeddings")
# w2v = g.Word2Vec()
w2v = g.Word2Vec.load_word2vec_format(sys.argv[4], binary=False)

#process data
if sys.argv[3] == "True" or sys.argv[3] == "T" or sys.argv[3] == "true" or sys.argv[3] == "t":
    # global data
    data = d.Data(filepath=sys.argv[1], lineSeparated=True)
else:
    # global data
    data = d.Data(filepath=sys.argv[1])


data.startServer()

#get size of vectors
w2vSize = len(w2v["the"])

#max length
maxLength = int(sys.argv[5])




#open file to estimate vocabulary size
f = open(sys.argv[1], "rb")

print("estimating vocabulary size")
#estimate vocabulary size
counter = Counter()

#max number of line to take from input file
max = int(sys.argv[2])
#counter to keep track of lines taken from file
c = 0

#build an estimation of the vocabulary size
for line in f:
    c+=1
    if c <= max:
        clean = line.rstrip()
        tokens = clean.split(" ")
        for t in tokens:
            counter[t] += 1

#close file
f.close()

#build the LSTM

#build the LSTM with static (not updated during training) vectors
model = Sequential()

#hyperparameters
w2v_dimension = w2vSize
max_sentence_length = maxLength
# voc_size = len(w2v.index2word)            #can it even generate words that weren't in training?
# voc_size = len(data.vocIDXtoLemma.keys()) #since building as we go, can't know this yet
voc_size = len(counter.items())         #over-estimation of vocabulary size; will this affect performance?
c_size = 100
num_epochs = 2
test_set = []

print("building LSTM")
#masking layer to handle variable lengths
    #all items are padded to max_sentence_length
    #batch_input_shape must be provided to first layer
model.add(Masking(mask_value=np.zeros(w2v_dimension), batch_input_shape=(1,1,w2v_dimension)))
#lstm layer
    #shape = batch size of 1, and update weights after each word
    #batch_input_shape was provided at first (Masking) layer
model.add(LSTM(output_dim=c_size, return_sequences=False, stateful=True, batch_input_shape=(1,1,w2v_dimension)))
#output layer to predict next word
# model.add(TimeDistributedDense(input_shape=c_size, output_dim=voc_size, activation="softmax"))
model.add(Dense(output_dim=voc_size, activation="softmax"))

# model.compile(loss="binary_crossentropy", optimizer="rmsprop")
# model.compile(loss="mse", optimizer="rmsprop")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary()

#open file to handle line at a time
f = open(sys.argv[1], "rb")

#training
print("training")

allWordVectorsPadded = []
allLemmaVectorsPadded = []

for i in range(num_epochs):
    #for first iteration, must process text first
    if i == 0:
        #counter to keep track of number of lines to process
        c = 0

        print("epoch", str(i+1))
        #counter to keep track of items to remove for testing
        l = 0
        #counter of words added to indices
        cWord = 0
        lWord = 0
        #iterate through each line
            #because training file is too long to do all at once
        for line in f:

            c += 1

            if c <= max_sentence_length:
                l += 1
                #skip any line that is larger than max length as set in command line
                if len(line.rstrip().split(" ")) < max_sentence_length:
                    #annotate line
                        #add to data.rawSents
                    data.annotateText(line)
                    #tokenize most recent sentence
                        #capture current idx for indexing
                    cWord, lWord = data.getTokenized(data.rawSents[-1], cWord, lWord)

                    print("current sentence in words", data.seqWords[-1])
                    print("current sentence in lemmas", data.seqLemmas[-1])

                    #convert most recent sentence to vector
                    wordVector = pre.convertSentenceToVec(data.seqWords[-1], w2v, w2vSize)
                    lemmaVector = pre.convertSentenceToVec(data.seqLemmas[-1], w2v, w2vSize)

                    #pad
                    print("padding")
                    wordVectorPadded = pre.padToConstant(wordVector, w2vSize, maxLength)
                    lemmaVectorPadded = pre.padToConstant(lemmaVector, w2vSize, maxLength)

                    #keep a 10th of the examples testing
                    if l % (max/10) == 0:
                        test_set.append(lemmaVectorPadded)
                    #otherwise, run through LSTM as training example
                    else:
                        #add to collection (for use in later epochs)
                        allWordVectorsPadded.append(wordVectorPadded)
                        allLemmaVectorsPadded.append(lemmaVectorPadded)
                        #loop through each word in the sequence, with an X of current word (j) and y of next word (j + 1)
                        for j in range(max_sentence_length-1):
                            jPlus1 = w2v.most_similar(positive=[lemmaVectorPadded[j+1]], topn=1)[0][0]
                            #bail on sentence when the next word is np.zeros
                            #either because it's padding or not in the W2V vectors
                            if np.all(lemmaVectorPadded[j+1] != np.zeros(w2v_dimension)):
                                print("epoch", str(i+1))
                                print("time step", j)
                                print("shape of vector at j", lemmaVectorPadded[j].shape)
                                # print("vector at j + 1", lemmaVectorPadded[j+1])
                                print("word at j + 1", jPlus1)
                                print("index at j + 1", data.vocLemmaToIDX[jPlus1])
                                # print("shape of vector at j + 1", lemmaVectorPadded[j+1].shape)
                                gold = np.zeros(voc_size)
                                gold[data.vocLemmaToIDX[jPlus1]] = 1.0
                                # print("one hot for j + 1", gold)
                                print("one hot for j + 1 shape", gold.shape)
                                model.train_on_batch(lemmaVectorPadded[j].reshape((1,1,w2v_dimension)), gold.reshape((1,voc_size)), accuracy=True)
                            else:
                                break
                        #at the end of the sequence reset the states
                        model.reset_states()
            else:
                break

#for all subsequent epochs, when data is already processed
    else:
        #pickle training and testing data
        if len(sys.argv) == 8:
            out = open(sys.argv[6], "wb")
            pickle.dump(allLemmaVectorsPadded, out)
            out.close()
            out2 = open(sys.argv[7], "wb")
            pickle.dump(test_set, out2)
            out2.close()
        #shuffle sentences at beginning of each epoch
        shuffle(allWordVectorsPadded)
        shuffle(allLemmaVectorsPadded)

        #iterate through all training sentences
        for sent in allLemmaVectorsPadded:
            #loop through each word in the sequence, with an X of current word (j) and y of next word (j + 1)
            for j in range(max_sentence_length-1):
                jPlus1 = w2v.most_similar(positive=[sent[j+1]], topn=1)[0][0]
                #bail on sentence when the next word is np.zeros
                #either because it's padding or not in the W2V vectors
                if np.all(sent[j+1] != np.zeros(w2v_dimension)):
                    print("epoch", str(i+1))
                    print("time step", j)
                    # print("training item shape", str(sent[j].shape))
                    # print("vector at j + 1", sent[j+1])
                    # print("shape of vector at j + 1", sent[j+1].shape)
                    print("word at j + 1", jPlus1)
                    print("index at j + 1", data.vocLemmaToIDX[jPlus1])
                    gold = np.zeros(voc_size)
                    gold[data.vocLemmaToIDX[jPlus1]] = 1.0
                    # print("one hot for j + 1", gold)
                    print("one hot for j + 1 shape", gold.shape)
                    model.train_on_batch(sent[j].reshape((1,1,w2v_dimension)), gold.reshape((1,voc_size)), accuracy=True)
                else:
                    break
            #at the end of the sequence reset the states
            model.reset_states()


#testing
print("testing")
#iterate through testing samples
allResults = []
for test_item in test_set:
    #list to keep accumulated sentence
    sentence = []
    #list to keep sentence results
    results = []
    #iterate through each word predicting the next word
    for m in range(max_sentence_length-1):
        #bail on sentence when the next word is np.zeros
        #either because it's padding or not in the W2V vectors
        if np.all(test_item[m] != np.zeros(w2v_dimension)):
            #distribution predicted from softmax
            distribution = model.predict_on_batch(test_item[m].reshape(1,1,w2v_dimension))
            # print("softmax output", distribution)
            #get index of most likely
            idx = np.argmax(distribution)
            #get closest word
            closest_word = data.vocIDXtoLemma[idx]
            #real next word
            real_word = w2v.most_similar(positive=[test_item[m+1]], topn=1)[0][0]
            sentence.append(closest_word)
            print("predicted sentence so far", sentence)
            print("predicted next word", closest_word)
            print("actual next word", real_word)
            if real_word == closest_word:
                results.append(1)
            else:
                results.append(0)
        else:
            break
    model.reset_states()
    sentenceAccuracy = float(results.count(1) / float(len(results)))
    print("final sentence", sentence)
    print("accuracy", str(sentenceAccuracy))
    allResults.append(sentenceAccuracy)

#final average accuracy
if len(allResults) == 0.0:
    averageAccuracy = 0
else:
    averageAccuracy = sum(allResults) / float(len(allResults))
print("final accuracy", str(averageAccuracy))

f.close()