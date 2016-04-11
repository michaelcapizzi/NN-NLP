from gensim import models as g
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *

#sys.argv[1] = file to process for data
#sys.argv[2] = line separated?
#sys.argv[3] = word2vec file

#get embeddings
print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)
#get size of vectors
w2vSize = len(w2v["the"])

#process data
if sys.argv[2] == "True" or sys.argv[2] == "T" or sys.argv[2] == "true" or sys.argv[2] == "t":
    data = d.Data(filepath=sys.argv[1], lineSeparated=True)
else:
    data = d.Data(filepath=sys.argv[1])

data.startServer()
print("annotating text")
data.annotateText()
data.getTokenized()


#convert to vectors
print("converting sentences to vectors")
# wordVectors = [pre.getVector(word, w2v, w2vSize) for word in data.seqWords[i] for i in range(len(data.seqWords))]
wordVectors = [pre.convertSentenceToVec(sentence, w2v, w2vSize) for sentence in data.seqWords]
# lemmaVectors = [pre.getVector(lemma, w2v, w2vSize) for lemma in data.seqLemmas[i] for i in range(len(data.seqLemmas))]
lemmaVectors = [pre.convertSentenceToVec(sentence, w2v, w2vSize) for sentence in data.seqLemmas]

#sort sequences into batches of same length
wordVectorsBatched = pre.sortBySeqLength(wordVectors)
lemmaVectorsBatched = pre.sortBySeqLength(lemmaVectors)

print("words")
print(wordVectors[0])
print("lemmas")
print(lemmaVectors[0])
print(wordVectorsBatched.keys())
print(lemmaVectorsBatched.keys())

#build the LSTM with static (not updated during training) vectors
model = Sequential()

#hyperparameters
w2v_dimension = w2vSize
max_sentence_length = max(lemmaVectorsBatched.keys())
voc_size = len(w2v.index2word)
c_size = w2vSize / 2

#masking layer to handle variable lengths
model.add(Masking(mask_value=0.0))
#lstm layer
    #shape (1,max_sentence_length,w2v_dimension)
model.add(LSTM(output_dim=c_size, return_sequences=True, stateful=True, batch_input_shape=(1,max_sentence_length,w2v_dimension)))
#output layer to predict next word
model.add(Dense(voc_size, activation="softmax"))


model.compile(loss="binary_crossentropy", optimizer="rmsprop")
