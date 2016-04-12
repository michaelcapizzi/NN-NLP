from gensim import models as g
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
from multiprocessing import Process

#sys.argv[1] = file to process for data
#sys.argv[2] = line separated?
#sys.argv[3] = word2vec file


#get embeddings
print("loading embeddings")
# w2v = g.Word2Vec()
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

#process data
if sys.argv[2] == "True" or sys.argv[2] == "T" or sys.argv[2] == "true" or sys.argv[2] == "t":
    # global data
    data = d.Data(filepath=sys.argv[1], lineSeparated=True)
else:
    # global data
    data = d.Data(filepath=sys.argv[1])


data.startServer()
print("annotating text")
data.annotateText()
data.getTokenized()

#get size of vectors
w2vSize = len(w2v["the"])

#convert to vectors
print("converting sentences to vectors")
wordVectors = [pre.convertSentenceToVec(sentence, w2v, w2vSize) for sentence in data.seqWords]
lemmaVectors = [pre.convertSentenceToVec(sentence, w2v, w2vSize) for sentence in data.seqLemmas]

#sort sequences into batches of same length
print("sorting by sequence length")
wordVectorsBatched = pre.sortBySeqLength(wordVectors)
lemmaVectorsBatched = pre.sortBySeqLength(lemmaVectors)
#pad
print("padding")
wordVectorsBatched = pre.padToLongest(wordVectorsBatched, w2vSize)
#pad
lemmaVectorsBatched = pre.padToLongest(lemmaVectorsBatched, w2vSize)

key = lemmaVectorsBatched.keys()[0]


#build the LSTM with static (not updated during training) vectors
model = Sequential()

#hyperparameters
w2v_dimension = w2vSize
max_sentence_length = max(lemmaVectorsBatched.keys())
# voc_size = len(w2v.index2word)            #can it even generate words that weren't in training?
voc_size = len(data.vocIDXtoLemma.keys())
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
# model.add(LSTM(input_dim=w2v_dimension, output_dim=c_size, return_sequences=True, stateful=True))
#output layer to predict next word
# model.add(TimeDistributedDense(input_shape=c_size, output_dim=voc_size, activation="softmax"))
model.add(Dense(output_dim=voc_size, activation="softmax"))


# model.compile(loss="binary_crossentropy", optimizer="rmsprop")
# model.compile(loss="mse", optimizer="rmsprop")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model.summary()

#training
print("training")
#set up manual training (to use train_on_batch)
for i in range(num_epochs):
    #list of different sequence lengths, shuffled
    shuffle(lemmaVectorsBatched.keys())
    #iterate through batches
    for k in lemmaVectorsBatched.keys():
        print("epoch", str(i+1), "sequence length", str(k))
        #get all training sentences from that sequence length
        samples = lemmaVectorsBatched[k]
        #add last example to testing set and train on all the rest
        if len(samples) > 1:
            for_training = samples[0:-1]
            test_set.append(samples[-1])
        else:
            for_training = samples
        #iterate through all training examples
        for x in for_training:
            #loop through each word in the sequence, with an X of current word (j) and y of next word (j + 1)
            for j in range(max_sentence_length-1):
                jPlus1 = w2v.most_similar(positive=[x[j+1]], topn=1)[0][0]
                #bail on sentence when the next word is np.zeros
                    #either because it's padding or not in the W2V vectors
                if np.all(x[j+1] != np.zeros(w2v_dimension)):
                    print("j", j)
                    # print("training item shape", str(x[j].shape))
                    print("vector at j + 1", x[j+1])
                    print("shape of vector at j + 1", x[j+1].shape)
                    print("word at j + 1", jPlus1)
                    print("index at j + 1", data.vocLemmaToIDX[jPlus1])
                    gold = np.zeros(voc_size)
                    gold[data.vocLemmaToIDX[jPlus1]] = 1.0
                    print("one hot for j + 1", gold)
                    # print("one hot for j + 1 shape", gold.shape)
                    model.train_on_batch(x[j].reshape((1,1,w2v_dimension)), gold.reshape((1,voc_size)), accuracy=True)
                else:
                    break
            #at the end of the sequence reset the states
            model.reset_states()
            # model.layers[1].reset_states()        #this is safer if I'm sure which layer is the LSTM


#testing
print("testing")
#iterate through testing samples
for test_item in test_set:
    #list to keep accumulated sentence
    sentence = []
    #list to keep results
    results = []
    #iterate through each word predicting the next word
    for m in range(max_sentence_length-1):
        #distribution predicted from softmax
        distribution = model.predict_on_batch(test_item[m].reshape(1,1,w2v_dimension))
        #get index of most likely
        idx = np.argmax(distribution)
        #get closest word
        closest_word = data.vocIDXtoLemma[idx]
        #real next word
        real_word = w2v.most_similar(positive=[test_item[m+1]], topn=1)
        sentence.append(closest_word)
        print("given: ", sentence)
        print("predicted: ", closest_word)
        print("actual: ", real_word)
        if real_word == closest_word:
            results.append(1)
        else:
            results.append(0)
    print("final sentence: ", sentence)
    print("accuracy: ", str(float(results.count(1)) / float(len(results))))



