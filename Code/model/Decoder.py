import tensorflow as tf
import numpy as np
from model.oneDecoderLayer import Decoder
class decoderBuilder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numDecoder = 6,Dv = 64,Dk = 256,nHead = 8):
        super().__init__()
        self.numDecoder = numDecoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim
    def build(self,inputShape):
        self.posEncoding = self.posEncoder(inputShape=inputShape[0])
        self.decoderLayers = [Decoder(denseDim = self.denseDim,Dv = self.Dv,Dk = self.Dk,nHead = self.nHead) for _ in range(self.numDecoder)]
    def call(self,inputs):
        y = inputs[0] + self.posEncoding
        x = self.decoderLayers[0]([y,inputs[1]])
        for dec in self.decoderLayers:
            x = dec([x,inputs[1]])
        return x
    def posEncoder(self,inputShape):
        posEncoding = np.zeros((inputShape[1], self.Dk))
        for k in range(inputShape[1]):
            for i in np.arange(int(self.Dk/2)):
                denominator = np.power(10000, 2*i/self.Dk)
                posEncoding[k, 2*i] = np.sin(k/denominator)
                posEncoding[k, 2*i+1] = np.cos(k/denominator)
        return posEncoding