import tensorflow as tf
from oneEncoderLayer import Encoder
import numpy as np

class encoderBuilder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numEncoder = 6,Dv = 64,Dk = 256,nHead = 8):
        super().__init__()
        self.numEncoder = numEncoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim            # dimention of middle feed forward network    
    def build(self,inputShape):
        self.posEncoding = self.posEncoder(inputShape=inputShape)
        self.encoders = [Encoder(denseDim = self.denseDim,numEncoder = self.numEncoder,Dv = self.Dv,Dk = self.Dk,nHead = self.nHead) for _ in range(self.numEncoder)]
    def call(self,inputs):
        y = inputs + self.posEncoding
        x = self.encoders[0](y)
        for enc in self.encoders[1:]:
            x = enc(x)
        return x
    def posEncoder(self,inputShape):
        posEncoding = np.zeros((inputShape[1], self.Dk))
        for k in range(inputShape[1]):
            for i in np.arange(int(self.Dk/2)):
                denominator = np.power(10000, 2*i/self.Dk)
                posEncoding[k, 2*i] = np.sin(k/denominator)
                posEncoding[k, 2*i+1] = np.cos(k/denominator)
        return posEncoding