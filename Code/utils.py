import tensorflow as tf
import pandas as pd
import numpy as np

class utils:
    def __init__(self,dataDir):
        self.dataDir = dataDir
    def loadData(self):
        self.trainData = pd.read_csv(self.dataDir + 'train.csv')
        self.testData = pd.read_csv(self.dataDir + 'test.csv')
    @staticmethod
    def preprocess(data):
        data['article'] = data['article'].str.replace('[^\w\s]','')
        data['article'] = data['article'].str.replace('\([^)]*\)','')
        data['highlights'] = '_start_' + ' ' + data['highlights'].str.replace('[^\w\s]','') + ' ' + '__end__'
        tok = tf.keras.preprocessing.text.Tokenizer()
        tok.fit_on_texts(list(data['article'].astype(str))) 
        text = tok.texts_to_sequences(list(data['article'].astype(str)))
