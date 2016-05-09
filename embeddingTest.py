from gensim import models as g
import Models.Embedding as e
import Data as d
import sys
import Utils.PreProcessing as pre
from keras.models import *
from keras.layers import *
from random import shuffle
import pickle
from collections import Counter
from multiprocessing import Process
import Models.Model as m



print("randomized embedding layer")

eRandom = e.Embedding_keras(voc_size=10000)

print("building random initialized layer")
eRandom.build()

# input_array = np.random.randint(100000, size=(1,1))
#
# mod = Sequential()
#
# mod.add(eRandom.layer)
#
# mod.compile("rmsprop", "mse")
#
# output_array=mod.predict(input_array)



randomModel = m.LSTM_keras(
                            embeddingLayerClass=eRandom,
                            num_epochs=1
)
#
# print("preparing data file")
# randomModel.prepareData("cocaForLM.txt", 100)
#
# randomModel.buildModel()



# print("loading embeddings")
# w2v = g.Word2Vec.load_word2vec_format("w2v_Gigaword.txt.gz", binary=False)
#
# print("embedding initialized layer")
#
# eW2V_maskTrue = e.Embedding_keras(load_w2v=True, gensim_class=w2v, voc_size=100000)
#
# eW2V_maskTrue.build()
#
# w2vModel = Sequential()
#
# w2vModel.add(eW2V_maskTrue.layer)
#
# w2vModel.summary()