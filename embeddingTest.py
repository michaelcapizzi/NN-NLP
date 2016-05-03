from gensim import models as g
import Embeddings.Embedding as e
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
import pickle
from collections import Counter
from multiprocessing import Process

print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format("w2v_Gigaword.txt.gz", binary=False)

# print("randomized embedding layer")
#
# eRandom = e.Embedding_keras(voc_size=100000)
#
# print("building random initialized layer")
# eRandom.build()
#
# randomModel = Sequential()
#
# randomModel.add(eRandom.layer)
#
# randomModel.summary()


print("embedding initialized layer")

eW2V_maskTrue = e.Embedding_keras(load_w2v=True, gensim_class=w2v, voc_size=100000)

eW2V_maskTrue.build()

w2vModel = Sequential()

w2vModel.add(eW2V_maskTrue.layer)

w2vModel.summary()