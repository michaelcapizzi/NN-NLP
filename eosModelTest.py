from gensim import models as g
import Data as d
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
import pickle
from collections import Counter
import Utils.Evaluation as e
import Models.Model as m

print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format("w2v_Gigaword.txt.gz", binary=False)

model = m.LSTM_keras(embeddingClass=w2v, w2vDimension=len(w2v["the"]), optimizer="adagrad", num_epochs=1)

model.prepareData("cocaForLM.txt", 25)

model.buildModel()

model.train("cocaForLM.txt")
