from Encoder import Encoder as encoder
from Decoder import Decoder as decoder
from tensorflow.keras import layers as ksl
import tensorflow as tf
class model:
    def __init__(self,inputSize,latentDim = 256):
        self.inputSize = inputSize
        self.latentDim = latentDim
    def buildModel(self):
        encoderInput = ksl.Input(self.inputSize[0])
        x = ksl.Embedding(self.inputSize[0],
                        self.latentDim,
                        mask_zero=False)(encoderInput)
        x = ksl.BatchNormalization()(x)
        encoderOutput = encoder()(x)

        decoderInput = ksl.Input(shape = [None,])
        x  = ksl.Embedding(self.inputSize[1],
                           self.latentDim,
                           mask_zero = False)
        x = decoder()([x,encoderOutput])
        x = ksl.BatchNormalization()(x)
        x = ksl.Dense(self.inputSize[1],activation = 'softmax')(x)
        model = tf.keras.Model([encoderInput,decoderInput],x)
        return model
        
