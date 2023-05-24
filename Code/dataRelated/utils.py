import tensorflow as tf
from transformers import AutoTokenizer
import json
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
    def preprocess(self,data):
        data['article'] = data['article'].str.replace('[^\w\s]','')
        data['article'] = data['article'].str.replace('\([^)]*\)','')
        decoderInput =  '_start_' + ' ' + data['highlights'].str.replace('[^\w\s]','')
        decoderOutput = data['highlights'].str.replace('[^\w\s]','') + ' ' + '__end__'
        data['highlights'] = '_start_' + ' ' + data['highlights'].str.replace('[^\w\s]','') + ' ' + '__end__'
        
        tok = tf.keras.preprocessing.text.Tokenizer()
        tok.fit_on_texts(list(data['article'].astype(str))) 
        text = tok.texts_to_sequences(list(data['article'].astype(str)))
        encoderVocabSize = len(tok.word_index)
        text = tf.keras.preprocessing.sequence.pad_sequences(text)
        tok = tf.keras.preprocessing.text.Tokenizer() 
        tok.fit_on_texts(list(data['highlights'].astype(str)))
        decoderInput = tok.texts_to_sequences(list(decoderInput))
        decoderInput = tf.keras.preprocessing.sequence.pad_sequences(decoderInput)
        
        decoderOutput = tok.texts_to_sequences(list(decoderOutput))
        decoderOutput = tf.keras.preprocessing.sequence.pad_sequences(decoderOutput)

        decoderVocabSize = len(tok.word_index) + 1
        return [text,decoderInput,decoderOutput,decoderVocabSize,encoderVocabSize]