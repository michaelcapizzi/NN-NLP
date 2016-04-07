from processors import *

"""
Currently not using preprocessing methods available in keras.  Using pyprocessors [https://github.com/myedibleenso/py-processors]
"""

#initialize a processor
def initializeProcessor():
    return Processor(port=8886)


#start server
def startServer(proc):
    proc.start_server(jar_path="processors-server.jar")


#annotate text
def annotate(proc, text):
    return proc.annotate(text)


"""
other methods
"""

#sort sequences by length
def sortBySeqLength(seqs):
    #instantiate empty dictionary
    dict = {}

    for s in seqs:
        #get length
        length = len(s)
        #if length alreayd in keys
        if length in dict.keys():
            #append this seq to list
            dict[length].append(s)
        else:
            #add key and this sequence as first item
            dict[length] = [s]

    return dict









