import tensorflow as tf
from multiheadAttention import multiheadAttention 
from tensorflow.keras import layers as ksl
class Encoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numEncoder = 6,Dv = 64,Dk = 512,nHead = 8):
        super().__init__()
        self.numEncoder = numEncoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim            # dimention of middle feed forward network    
    def build(self,inputShape):
        self.attention = multiheadAttention(Dk = self.Dk,Dv = self.Dv,nHead=self.nHead)
    def call(self,inputs):
        y = inputs
        for _ in range(self.numEncoder):
            x = self.attention([y,y,y])
            x =  ksl.Add()([y,x])
            x =  ksl.LayerNormalization()(x)
            ff = ksl.Dense(self.denseDim,activation = 'relu')(x)
            ff = ksl.Dense(inputs.shape[-1])(ff)
            x =  ksl.Add()([ff,x])
            y =  ksl.LayerNormalization()(x)
        return y
