import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from transformers import AutoTokenizer
import json
import pandas as pd
class utils:
    def __init__(self,dataAddr):
        self.dataAddr = dataAddr
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data = []
    def loadData(self):
        with open(self.dataAddr,'r') as f:
            data = json.load(f)
        for k in data.keys():
            self.data.append(data[k]['corpus'])
        self.data = pd.DataFrame(self.data,columns=['corpus'])
    def preprocess(self,data):
        data['corpus'] = data['corpus'].str.replace('[^\w\s]','')
        data['corpus'] = data['corpus'].str.replace('\([^)]*\)','')
        decoderInput =  '_start_' + ' ' + data['corpus']
        decoderOutput = data['corpus'] + ' ' + '__end__'
        data['corpus'] = '_start_' + ' ' + data['corpus']+ ' ' + '_end_'
        tok = tf.keras.preprocessing.text.Tokenizer() 
        tok.fit_on_texts(list(data['corpus'].astype(str)))
        decoderInput = tok.texts_to_sequences(list(decoderInput))
        decoderInput = tf.keras.preprocessing.sequence.pad_sequences(decoderInput)
        decoderOutput = tok.texts_to_sequences(list(decoderOutput))
        decoderOutput = tf.keras.preprocessing.sequence.pad_sequences(decoderOutput)
        decoderVocabSize = len(tok.word_index) + 1
        return [decoderInput,decoderOutput,decoderVocabSize]