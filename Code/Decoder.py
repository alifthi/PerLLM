import tensorflow as tf
from Encoder import Encoder as encoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numDecoder = 6,Dv = 64,Dk = 256,nHead = 8):
        super().__init__()
        self.numDecoder = numDecoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim
    def build(self,inputShape):
        from multiheadAttention import multiheadAttention 
        self.maskedAttention = multiheadAttention(Dk = self.Dk,Dv = self.Dv,
                                                    nHead=self.nHead,masked=True)
        self.attention = multiheadAttention(Dk = self.Dk,Dv = self.Dv,
                                                    nHead=self.nHead)
        self.posEncoding = self.posEncoder(inputShape=inputShape[1])
    def call(self,inputs):
        from tensorflow.keras import layers as ksl
        y = inputs[0] + self.posEncoding
        for _ in range(self.numDecoder):
            x = self.maskedAttention([y,y,y])
            x =  ksl.Add()([y,x])
            x =  ksl.LayerNormalization()(x)
            encoderOut = inputs[1] # self.encoder(inputs[1])
            attention = self.attention([x,encoderOut,encoderOut])
            x =  ksl.Add()([attention,x])
            x =  ksl.LayerNormalization()(x)
            ff = ksl.Dense(self.denseDim,activation = 'relu')(x)
            ff = ksl.Dense(y.shape[-1])(ff)
            x =  ksl.Add()([ff,x])
            y =  ksl.LayerNormalization()(x)
        return y
    def posEncoder(self,inputShape):
        import numpy as np
        posEncoding = np.zeros((inputShape[1], self.Dk))
        for k in range(inputShape[1]):
            for i in np.arange(int(self.Dk/2)):
                denominator = np.power(10000, 2*i/self.Dk)
                posEncoding[k, 2*i] = np.sin(k/denominator)
                posEncoding[k, 2*i+1] = np.cos(k/denominator)
        return posEncoding