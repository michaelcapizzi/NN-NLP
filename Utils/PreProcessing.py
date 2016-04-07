from processors import *

"""
Currently not using preprocessing methods available in keras.  Using pyprocessors [https://github.com/myedibleenso/py-processors]
"""

#initialize a processor
def initializeProcessor():
    Processor(port=8886)


#start server
def startServer(proc):
    proc.start_server(jarpath="processors-server.jar", port=8886)


#annotate text
def annotate(proc, text):
    proc.annotate(text)






