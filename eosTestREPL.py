from gensim import models as g
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
import pickle
from collections import Counter
import Utils.Evaluation as e
from multiprocessing import Process

#TODO look here! https://github.com/fchollet/keras/issues/395
#TODO look here! http://blog.thedigitalcatonline.com/blog/2013/03/26/python-generators-from-iterators-to-cooperative-multitasking-2/#.VxacbB8zrCI

#get embeddings
print("loading embeddings")
# w2v = g.Word2Vec()
w2v = g.Word2Vec.load_word2vec_format("w2v_Gigaword.txt.gz", binary=False)

data = d.Data(filepath="cocaForLM.txt", lineSeparated=True)

data.startServer()

#get size of vectors
w2vSize = len(w2v["the"])

#max length
maxLength = 30

#max number of line to take from input file
max = 10

#build the LSTM

#build the LSTM with static (not updated during training) vectors
model = Sequential()

#hyperparameters
w2v_dimension = w2vSize
max_sentence_length = maxLength
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
# model.add(Dense(output_dim=voc_size, activation="softmax"))
model.add(Dense(output_dim=2, activation="softmax"))            #labels: EOS or nonEOS

# model.compile(loss="binary_crossentropy", optimizer="rmsprop")
# model.compile(loss="mse", optimizer="rmsprop")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model.summary()

#open file to handle line at a time
f = open("cocaForLM.txt", "rb")

#training
print("training")

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

            #skip any line that is larger than max length as set in command line
            if len(line.rstrip().split(" ")) < max_sentence_length:
                #annotate line
                #add to data.rawSents
                data.annotateText(line)
                #tokenize most recent sentence
                #capture current idx for indexing
                cWord, lWord = data.getTokenized(data.rawSents[-1], cWord, lWord)

                print("current sentence in lemmas", data.seqLemmas[-1])

                #convert most recent sentence to vector
                # wordVector = pre.convertSentenceToVec(data.seqWords[-1], w2v, w2vSize)
                lemmaVector = pre.convertSentenceToVec(data.seqLemmas[-1], w2v, w2vSize)

                #only keep sentences that have all words in word2vec list
                if np.all(lemmaVector != np.zeros(w2v_dimension)):

                    c += 1

                    if c <= max_sentence_length:
                        l += 1

                        #pad
                        print("padding")
                        # wordVectorPadded = pre.padToConstant(wordVector, w2vSize, maxLength)
                        lemmaVectorPadded = pre.padToConstant(lemmaVector, w2vSize, maxLength)

                        #keep a 10th of the examples testing
                        if l % (max/10) == 0:
                            test_set.append(lemmaVectorPadded)
                        #otherwise, run through LSTM as training example
                        else:
                            #add to collection (for use in later epochs)
                            # allWordVectorsPadded.append(wordVectorPadded)
                            allLemmaVectorsPadded.append(lemmaVectorPadded)
                            #loop through each word in the sequence, with an X of current word (j) and y of EOS or non_EOS
                            for j in range(max_sentence_length-1):
                                print("epoch", str(i+1))
                                print("time step", j)
                                print("shape of vector at j", lemmaVectorPadded[j].shape)
                                #if the current word is the last in the sentence or the next word is padding or the end of the sequence, gold is EOS
                                if j == max_sentence_length - 1 or np.all(lemmaVectorPadded[j+1] == np.zeros(w2v_dimension)):
                                    gold = np.array([1,0])
                                else:
                                    gold = np.array([0,1])
                                print("label", gold)
                                model.train_on_batch(lemmaVectorPadded[j].reshape(1,1,w2v_dimension), gold.reshape(1,2), accuracy=True)
                            #at the end of the sequence reset the states
                            model.reset_states()
                    else:
                        break
        f.close()
    #for all subsequent epochs, when data is already processed
    else:
        #shuffle sentences at beginning of each epoch
        shuffle(allLemmaVectorsPadded)

        #iterate through all training sentences
        for sent in allLemmaVectorsPadded:
            #loop through each word in the sequence, with an X of current word (j) and y of next word (j + 1)
            for j in range(max_sentence_length-1):
                print("epoch", str(i+1))
                print("time step", j)
                print("shape of vector at j", lemmaVectorPadded[j].shape)
                #if the current word is the last in the sentence or the next word is padding or the end of the sequence, gold is EOS
                if j == max_sentence_length - 1 or np.all(lemmaVectorPadded[j+1] == np.zeros(w2v_dimension)):
                    gold = np.array([1,0])
                else:
                    gold = np.array([0,1])
                print("label", gold)
                model.train_on_batch(lemmaVectorPadded[j].reshape(1,1,w2v_dimension), gold.reshape(1,2), accuracy=True)
                #at the end of the sequence reset the states
            model.reset_states()

#testing
print("testing")
#iterate through testing samples
allResults = []
# for test_item in test_set:
#     #list to keep accumulated sentence
#     sentence = []
#     #list to keep sentence results
#     results = []
#     #iterate through each word predicting the next word
#     for m in range(max_sentence_length-1):
#         # distribution = model.predict_on_batch(test_item[m].reshape(1,1,w2v_dimension))
#         #get softmax of labels
#         distribution = model.predict_on_batch(test_item[m].reshape((1,1,w2v_dimension)))
#         #get argmax of softmax
#         label = np.argmax(distribution)
#         #get actual
#         if m == max_sentence_length - 1 or np.all(test_item[m+1] == np.zeros(w2v_dimension)):
#             actual = 1
#         else:
#             actual = 0
#         #get the word associated with the vector
#         word = w2v.most_similar(positive=[test_item[m]], topn=1)[0][0]
#         #add to sentence
#         sentence.append(word)
#         print("current sentence", sentence)
#         print("predicted", label)
#         print("actual", actual)
#         #record results
#         if actual == 1 and label == 1:
#             results.append("tp")
#             print("end of sentence")
#             break
#         elif actual == 1 and label == 0:
#             results.append("fn")
#             print("end of sentence")
#             break
#         elif actual == 0 and label == 1:
#             results.append("fp")
#         else:
#             results.append("tn")
#     #reset state of net
#     model.reset_states()
#     #count true / false negative/positives
#     tp = results.count("tp")
#     tn = results.count("tn")
#     fp = results.count("fp")
#     fn = results.count("fn")
#     precision = e.precision(tp, fp)
#     recall = e.recall(tp, fn)
#     f1 = e.f1(precision, recall)
#     print("sentence precision", precision)
#     print("sentence recall", recall)
#     print("sentence f1", f1)
#     #add all results to final results
#     for r in results:
#         allResults.append(r)
#
# #final results
# finalTP = allResults.count("tp")
# finalTN = allResults.count("tn")
# finalFP = allResults.count("fp")
# finalFN = allResults.count("fn")
# finalPrecision = e.precision(finalTP, finalFP)
# finalRecall = e.recall(finalTP, finalFN)
# finalF1 = e.f1(finalPrecision, finalRecall)
# print("final precision", finalPrecision)
# print("final recall", finalRecall)
# print("final f1", finalF1)
#
